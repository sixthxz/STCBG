"""
merged_basin_validation_pipeline.py  (v2 — with closure operator)
──────────────────────────────────────────────────────────────────────────────
Changes vs original
───────────────────
1. compute_basin_cpu: np.roll → circular slot index (no-alloc buffer)
2. NCC loop: ncc_surfaces dict populated in-loop (one extra line)
3. closure_operator() called after the NCC loop — new layer
4. genuine_pair_certificate replaces the implicit "best NCC is enough"
5. Visualization: Panel 5 → partition entropy curve
                  Panel 6 → class score distribution
   (drift trajectory moved to standalone save if needed)
6. Summary block extended with closure metrics
7. ckpt integration for all artifacts (optional — activate by setting
   USE_CHECKPOINT = True and pointing CKPT_ROOT to your Drive folder)

Requires: phase_recovery  (in scope or phase_recovery.npy on disk)
"""

import gc
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from typing import Optional

from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image as PILImage


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
rng  = np.random.default_rng(SEED)

SAVEKW = dict(dpi=220, bbox_inches="tight")

# Basin
XMIN, XMAX = -2.0, 2.0
COUPLING   = 0.92
FEEDBACK   = 0.12
W_BASIN    = 20
STEPS      = 200
BASIN_RES  = 800    # [⚠ VRAM] see rrg_basin_gpu.py for limits

# NCC
SCALES = np.linspace(0.05, 0.60, 30)

# Monte Carlo stability
N_TRIALS = 25
MC_NOISE = 0.10

# Peak analysis
PEAK_WINDOW_RADIUS = 20
FOOTPRINT          = 25
TOPK               = 10

EPS = 1e-8

# ── Closure operator tuning ───────────────────────────────────────────────────
# These are the only new parameters vs the original pipeline.
# Defaults work for BASIN_RES=800, 30 scales, NCC in [0.5, 0.9].
CLOSURE_EPSILON_PX          = 5.0    # [⚠ tune] spatial tolerance (pixels)
                                      #   too small → many singleton classes
                                      #   too large → merges distinct matches
CLOSURE_EPSILON_SCORE       = 0.02   # [⚠ tune] NCC score tolerance per class
CLOSURE_EPSILON_ENTROPY     = 0.005  # [⚠ tune] ΔH threshold for closure
                                      #   raise to 0.01 if closure never fires
CLOSURE_THRESHOLD_STABILITY = 0.015  # [⚠ tune] max σ within dominant class
                                      #   for genuine_pair_certificate = True
CLOSURE_MIN_SCALES          = 5      # guard: don't close before 5 scales seen

# ── Optional checkpoint ───────────────────────────────────────────────────────
USE_CHECKPOINT = True    # CKPT_DIR y FIGURES_DIR vienen de Celda 00
# Si se corre standalone (sin Celda 00), definir manualmente:
# CKPT_DIR    = Path("/content/drive/MyDrive/STCBG/output/datasets")
# FIGURES_DIR = Path("/content/drive/MyDrive/STCBG/output/figures")


# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════

try:
    phase_recovery
except NameError:
    phase_recovery = np.load("phase_recovery.npy")

phase_recovery = np.asarray(phase_recovery, dtype=np.float32)
print(f"Template range: [{phase_recovery.min():.4f}, {phase_recovery.max():.4f}]")

if USE_CHECKPOINT:
    from rrg_checkpoint import CheckpointStore
    ckpt = CheckpointStore(CKPT_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def normalize(a):
    a   = np.asarray(a, dtype=np.float32)
    std = a.std()
    return np.zeros_like(a) if std < EPS else (a - a.mean()) / (std + EPS)


def compute_ncc_surface(basin_norm, template_array, scale):
    th = max(int(template_array.shape[1] * scale), 8)
    tw = max(int(template_array.shape[0] * scale), 8)

    if th >= basin_norm.shape[0] or tw >= basin_norm.shape[1]:
        return None

    template_resized = np.array(
        PILImage.fromarray(template_array.T.astype(np.float32))
                .resize((tw, th), PILImage.BILINEAR),
        dtype=np.float32,
    )
    t_norm       = normalize(template_resized)
    template_std = t_norm.std()

    if template_std < EPS:
        return None

    corr  = fftconvolve(basin_norm, t_norm[::-1, ::-1], mode="valid")
    denom = th * tw * max(basin_norm.std(), EPS) * max(template_std, EPS)
    ncc   = corr / denom

    max_idx = np.unravel_index(np.argmax(ncc), ncc.shape)
    score   = float(ncc[max_idx])

    return {"ncc": ncc, "score": score, "idx": max_idx,
            "shape": (th, tw), "template": template_resized}


# ══════════════════════════════════════════════════════════════════════════════
# BASIN — circular-buffer version (no np.roll alloc)
# ══════════════════════════════════════════════════════════════════════════════

def compute_basin_cpu(xmin, xmax, res, coupling, feedback, W, steps):
    coords = np.linspace(xmin, xmax, res)
    X, Y   = np.meshgrid(coords, coords)
    x, y   = X.copy(), Y.copy()

    # [⚠ VRAM / RAM] 3 × W × res² × 4 B
    #   res=800, W=20 → ~158 MB  (safe on any modern machine)
    x_hist   = np.zeros((W, res, res), dtype=np.float32)
    y_hist   = np.zeros((W, res, res), dtype=np.float32)
    rho_hist = np.zeros((W, res, res), dtype=np.float32)

    basin      = np.ones((res, res),  dtype=np.float32)
    classified = np.zeros((res, res), dtype=bool)
    t0         = time.time()

    for t in range(steps):
        x_new = np.sin(coupling * y) + feedback * x
        y_new = np.sin(coupling * x) - feedback * y
        x, y  = x_new, y_new

        # Circular slot — avoids np.roll full-buffer copy each step
        slot         = t % W
        x_hist[slot] = x
        y_hist[slot] = y

        if t >= W:
            xm  = x_hist - x_hist.mean(axis=0, keepdims=True)
            ym  = y_hist - y_hist.mean(axis=0, keepdims=True)
            num = (xm * ym).sum(axis=0)
            den = np.sqrt((xm ** 2).sum(axis=0) * (ym ** 2).sum(axis=0))
            rho = np.where(den > EPS, num / den, 0.0)
            rho_hist[slot] = rho

        if t > 40 and t >= W:
            local_std = rho_hist.std(axis=0)
            dr        = rho_hist.var(axis=0)
            rho_mean  = rho_hist.mean(axis=0)
            dlt       = dr - ((1 - rho_mean ** 2) ** 2) / W

            stable  = (local_std < 0.002)                               & ~classified
            oscill  = (local_std >= 0.002) & (local_std < 0.03)         & ~classified
            release = (local_std >= 0.03)  & (dlt > 0.08)               & ~classified

            basin[stable]  = 0.0
            basin[oscill]  = 1.0
            basin[release] = 2.0
            classified    |= stable | oscill | release

            if classified.all():
                print(f"Early exit at t={t}")
                break

    print(f"Basin generated in {time.time()-t0:.1f}s")
    return basin


# ══════════════════════════════════════════════════════════════════════════════
# CLOSURE OPERATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClosureResult:
    closure_reached             : bool
    closure_scale               : Optional[float]
    n_classes                   : int
    dominant_class              : int
    class_sizes                 : np.ndarray
    partition_entropy_curve     : np.ndarray
    entropy_delta_curve         : np.ndarray
    class_score_mean            : np.ndarray
    class_score_std             : np.ndarray
    dominant_class_stability    : float
    genuine_pair_certificate    : bool
    summary                     : str = field(default="", repr=False)


def _build_partition(positions, scores, eps_px, eps_score):
    n      = len(scores)
    labels = np.full(n, -1, dtype=np.int32)
    ctr_p  = []
    ctr_s  = []
    nxt    = 0
    for i in range(n):
        hit = False
        for c, (cp, cs) in enumerate(zip(ctr_p, ctr_s)):
            if np.linalg.norm(positions[i] - cp) <= eps_px and abs(scores[i] - cs) <= eps_score:
                labels[i] = c
                hit = True
                break
        if not hit:
            labels[i] = nxt
            ctr_p.append(positions[i].copy())
            ctr_s.append(scores[i])
            nxt += 1
    return labels


def _H(labels):
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / len(labels)
    return float(-np.sum(p * np.log(p + 1e-12)))


def closure_operator(scores_log, ncc_surfaces, ppu,
                     epsilon_px=CLOSURE_EPSILON_PX,
                     epsilon_score=CLOSURE_EPSILON_SCORE,
                     epsilon_entropy=CLOSURE_EPSILON_ENTROPY,
                     threshold_stability=CLOSURE_THRESHOLD_STABILITY,
                     min_scales=CLOSURE_MIN_SCALES,
                     verbose=True) -> ClosureResult:
    """
    Collapses the multiscale NCC explanation space into equivalence classes
    under observational indistinguishability.

    Two scales are equivalent if:
      • their argmax positions differ by ≤ epsilon_px pixels
      • their NCC scores differ by ≤ epsilon_score

    Closure fires when ΔH(partition) < epsilon_entropy, meaning adding a new
    scale no longer creates a new distinguishable class.

    The RRG pair is certified genuine when:
      closure_reached AND dominant class is stable AND is the largest class.
    """
    scales = np.array([s for s, _ in scores_log])
    scores = np.array([sc for _, sc in scores_log])
    n      = len(scales)

    positions = np.zeros((n, 2), dtype=np.float32)
    for i, (s, _) in enumerate(scores_log):
        surf         = ncc_surfaces[s]
        positions[i] = np.unravel_index(np.argmax(surf), surf.shape)

    # Incremental entropy curve
    H_curve = np.zeros(n)
    for k in range(1, n + 1):
        lk       = _build_partition(positions[:k], scores[:k], epsilon_px, epsilon_score)
        H_curve[k - 1] = _H(lk)

    dH = np.abs(np.diff(H_curve))

    # Closure detection
    closure_scale   = None
    closure_reached = False
    for k in range(min_scales, n):
        if dH[k - 1] < epsilon_entropy:
            closure_reached = True
            closure_scale   = float(scales[k])
            break

    # Final full partition
    final_labels              = _build_partition(positions, scores, epsilon_px, epsilon_score)
    unique_c, counts          = np.unique(final_labels, return_counts=True)
    n_classes                 = len(unique_c)
    class_score_mean          = np.array([scores[final_labels == c].mean() for c in unique_c])
    class_score_std           = np.array([scores[final_labels == c].std()  for c in unique_c])

    global_best_idx           = int(np.argmax(scores))
    dominant_class            = int(final_labels[global_best_idx])
    dominant_ci               = int(np.where(unique_c == dominant_class)[0][0])
    dom_stability             = float(class_score_std[dominant_ci])
    largest_class             = int(unique_c[np.argmax(counts)])

    genuine = (closure_reached
               and dom_stability < threshold_stability
               and dominant_class == largest_class)

    lines = [
        "─" * 58,
        "CLOSURE OPERATOR — Observational Equivalence",
        "─" * 58,
        f"  Scales evaluated          : {n}",
        f"  Equivalence classes found : {n_classes}",
        f"  Class sizes               : {counts.tolist()}",
        f"  Dominant class            : C{dominant_class}  "
          f"(mean NCC={class_score_mean[dominant_ci]:.4f}, σ={dom_stability:.4f})",
        f"  Closure reached           : {closure_reached}",
        f"  Closure scale             : {closure_scale}",
        f"  Final partition H         : {H_curve[-1]:.5f} nats",
        "─" * 58,
        f"  Genuine pair certificate  : {'✓  TRUE' if genuine else '✗  FALSE'}",
        f"    closure_reached         : {closure_reached}",
        f"    dominant σ < {threshold_stability}        : {dom_stability < threshold_stability}  (σ={dom_stability:.4f})",
        f"    dominant == largest     : {dominant_class == largest_class}",
        "─" * 58,
    ]
    summary = "\n".join(lines)
    if verbose:
        print(summary)

    return ClosureResult(
        closure_reached=closure_reached, closure_scale=closure_scale,
        n_classes=n_classes, dominant_class=dominant_class,
        class_sizes=counts, partition_entropy_curve=H_curve,
        entropy_delta_curve=dH, class_score_mean=class_score_mean,
        class_score_std=class_score_std,
        dominant_class_stability=dom_stability,
        genuine_pair_certificate=genuine, summary=summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUN — Basin
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating basin map...")
basin      = (ckpt.load("basin_map") if USE_CHECKPOINT else None) or \
             compute_basin_cpu(XMIN, XMAX, BASIN_RES, COUPLING, FEEDBACK, W_BASIN, STEPS)

if USE_CHECKPOINT:
    ckpt.save("basin_map", basin)

basin_norm = normalize(basin)
ppu        = BASIN_RES / (XMAX - XMIN)


# ══════════════════════════════════════════════════════════════════════════════
# RUN — Multiscale NCC  (+ncc_surfaces dict for closure)
# ══════════════════════════════════════════════════════════════════════════════

print("\nSearching multiscale NCC...")
best        = None
scores_log  = []
ncc_surfaces = {}          # ← feeds the closure operator

for s in SCALES:
    result = compute_ncc_surface(basin_norm, phase_recovery, s)
    if result is None:
        continue
    score = result["score"]
    scores_log.append((s, score))
    ncc_surfaces[s] = result["ncc"]   # ← one extra line vs original

    if best is None or score > best["score"]:
        best = {"scale": s, **result}

    print(f"  scale={s:.3f}  NCC={score:.4f}", end="\r")

print()
if best is None:
    raise RuntimeError("No valid NCC match found.")

best_row, best_col = best["idx"]
bh, bw             = best["shape"]
mx0 = XMIN + best_col / ppu;  mx1 = mx0 + bw / ppu
my0 = XMIN + best_row / ppu;  my1 = my0 + bh / ppu

print(f"Best NCC={best['score']:.4f} @ scale={best['scale']:.3f}")
print(f"Domain: x=[{mx0:.3f},{mx1:.3f}]  y=[{my0:.3f},{my1:.3f}]")


# ══════════════════════════════════════════════════════════════════════════════
# RUN — Closure operator  ← NEW LAYER
# ══════════════════════════════════════════════════════════════════════════════

print("\nRunning closure operator...")
closure = closure_operator(scores_log, ncc_surfaces, ppu)

if USE_CHECKPOINT:
    ckpt.save_npz("closure_result",
        closure_reached         = np.bool_(closure.closure_reached),
        closure_scale           = np.float32(closure.closure_scale or -1),
        n_classes               = np.int32(closure.n_classes),
        dominant_class          = np.int32(closure.dominant_class),
        class_sizes             = closure.class_sizes,
        partition_entropy_curve = closure.partition_entropy_curve,
        entropy_delta_curve     = closure.entropy_delta_curve,
        class_score_mean        = closure.class_score_mean,
        class_score_std         = closure.class_score_std,
        dominant_stability      = np.float32(closure.dominant_class_stability),
        genuine                 = np.bool_(closure.genuine_pair_certificate),
    )


# ══════════════════════════════════════════════════════════════════════════════
# RUN — Peak topology
# ══════════════════════════════════════════════════════════════════════════════

print("\nAnalyzing NCC topology...")
ncc_base  = best["ncc"]
y_center, x_center = best["idx"]

y0_w = max(0, y_center - PEAK_WINDOW_RADIUS)
y1_w = min(ncc_base.shape[0], y_center + PEAK_WINDOW_RADIUS)
x0_w = max(0, x_center - PEAK_WINDOW_RADIUS)
x1_w = min(ncc_base.shape[1], x_center + PEAK_WINDOW_RADIUS)

peak_window        = ncc_base[y0_w:y1_w, x0_w:x1_w]
X_mesh, Y_mesh     = np.meshgrid(np.arange(x0_w, x1_w), np.arange(y0_w, y1_w))


# ══════════════════════════════════════════════════════════════════════════════
# RUN — Uniqueness
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing uniqueness metrics...")
local_max       = ncc_base == maximum_filter(ncc_base, size=FOOTPRINT)
candidate_idx   = np.argwhere(local_max)
cands           = sorted([(float(ncc_base[y, x]), int(y), int(x))
                           for y, x in candidate_idx], reverse=True)
top_peaks       = cands[:TOPK]
uniqueness_gap  = (top_peaks[0][0] - top_peaks[1][0]) if len(top_peaks) > 1 else top_peaks[0][0]
print(f"Uniqueness gap = {uniqueness_gap:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN — Stochastic drift
# ══════════════════════════════════════════════════════════════════════════════

print("\nRunning stochastic drift test...")
noise_levels      = [0.0, 0.05, 0.10, 0.20, 0.40]
drift_trajectory  = []
base_x_real       = XMIN + x_center / ppu
base_y_real       = XMIN + y_center / ppu

for noise in noise_levels:
    if noise == 0.0:
        drift_trajectory.append((base_x_real, base_y_real, best["score"]))
        continue
    noisy  = np.clip(phase_recovery + rng.normal(0, noise, phase_recovery.shape), 0.0, 2.0)
    result = compute_ncc_surface(basin_norm, noisy, best["scale"])
    ny, nx = result["idx"]
    drift_trajectory.append((XMIN + nx / ppu, XMIN + ny / ppu, result["score"]))
    print(f"  noise={noise:.2f}  NCC={result['score']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN — Reversibility
# ══════════════════════════════════════════════════════════════════════════════

print("\nTesting reversibility...")
t_up      = np.clip(phase_recovery + rng.normal(0, 0.40, phase_recovery.shape), 0, 2)
t_relax   = np.clip(phase_recovery + rng.normal(0, 0.02, phase_recovery.shape), 0, 2)
r_relax   = compute_ncc_surface(basin_norm, t_relax, best["scale"])
ry, rx    = r_relax["idx"]
recovery_distance = np.sqrt((XMIN + rx/ppu - base_x_real)**2 +
                             (XMIN + ry/ppu - base_y_real)**2)
print(f"Recovery distance = {recovery_distance:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN — Monte Carlo stability
# ══════════════════════════════════════════════════════════════════════════════

print("\nMonte Carlo stability audit...")
mc_coords, mc_scores = [], []
for _ in range(N_TRIALS):
    noisy  = np.clip(phase_recovery + rng.normal(0, MC_NOISE, phase_recovery.shape), 0, 2)
    result = compute_ncc_surface(basin_norm, noisy, best["scale"])
    yy, xx = result["idx"]
    mc_coords.append((XMIN + xx/ppu, XMIN + yy/ppu))
    mc_scores.append(result["score"])

mc_coords  = np.array(mc_coords)
mc_scores  = np.array(mc_scores)
coord_std  = mc_coords.std(axis=0)
score_std  = mc_scores.std()
print(f"Coordinate std = ({coord_std[0]:.5f}, {coord_std[1]:.5f})")
print(f"NCC std        = {score_std:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION  (6-panel report)
# ══════════════════════════════════════════════════════════════════════════════

print("\nRendering report...")

fig = plt.figure(figsize=(18, 10), facecolor="white")

# Panel 1 — Recovery template
ax1 = fig.add_subplot(231)
im1 = ax1.imshow(phase_recovery.T, origin="lower", aspect="auto",
                 cmap="viridis", vmin=0, vmax=2)
ax1.set_title("Recovery template")
plt.colorbar(im1, ax=ax1)

# Panel 2 — Basin + match box
ax2 = fig.add_subplot(232)
im2 = ax2.imshow(basin, origin="lower", cmap="viridis", vmin=0, vmax=2,
                 extent=[XMIN, XMAX, XMIN, XMAX])
ax2.add_patch(patches.Rectangle((mx0, my0), mx1-mx0, my1-my0,
              lw=2, edgecolor="red", facecolor="none", ls="--"))
cert_str = "✓ genuine" if closure.genuine_pair_certificate else "✗ open"
ax2.set_title(f"Best match  NCC={best['score']:.3f}\n"
              f"scale={best['scale']:.3f}  |  {cert_str}")

# Panel 3 — Zoom region
ax3 = fig.add_subplot(233)
zoom = basin[best_row:best_row+bh, best_col:best_col+bw]
im3  = ax3.imshow(zoom, origin="lower", cmap="viridis", vmin=0, vmax=2)
ax3.set_title("Zoom region")
plt.colorbar(im3, ax=ax3)

# Panel 4 — NCC peak topology (3D)
ax4 = fig.add_subplot(234, projection="3d")
ax4.plot_surface(X_mesh, Y_mesh, peak_window, cmap="magma", edgecolor="none")
ax4.set_title("NCC peak topology")
ax4.set_zlabel("NCC")

# Panel 5 — Partition entropy curve  ← replaces old "NCC vs scale"
ax5 = fig.add_subplot(235)
scales_arr = np.array([s for s, _ in scores_log])
ax5.plot(scales_arr, closure.partition_entropy_curve,
         "o-", ms=4, lw=1.5, color="#3B82F6", label="H(partition)")
if closure.closure_scale is not None:
    ax5.axvline(closure.closure_scale, color="red", ls="--", lw=1.2,
                label=f"closure @ {closure.closure_scale:.3f}")
ax5.set_xlabel("scale")
ax5.set_ylabel("partition entropy (nats)")
ax5.set_title(f"Equivalence closure\n"
              f"{'CLOSED' if closure.closure_reached else 'open'}  "
              f"| classes={closure.n_classes}")
ax5.legend(fontsize=8)

# Panel 6 — Class score distribution
ax6 = fig.add_subplot(236)
x_bar  = np.arange(closure.n_classes)
colors = ["#EF4444" if i == closure.dominant_class else "#6B7280"
          for i in range(closure.n_classes)]
ax6.bar(x_bar, closure.class_score_mean, yerr=closure.class_score_std,
        color=colors, alpha=0.85, capsize=4, width=0.6)
ax6.set_xticks(x_bar)
ax6.set_xticklabels(
    [f"C{i}\n(n={closure.class_sizes[i]})" for i in range(closure.n_classes)],
    fontsize=8)
ax6.set_ylabel("mean NCC score")
ax6.set_title(f"Equivalence classes\n"
              f"dom σ={closure.dominant_class_stability:.4f}")

plt.suptitle("Unified basin validation pipeline  —  RRG closure",
             fontsize=14, y=1.02)
plt.tight_layout()

fig.savefig("merged_basin_validation_pipeline.png", **SAVEKW)
fig.savefig("merged_basin_validation_pipeline.pdf", **SAVEKW)
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + closure.summary)
print(f"Best NCC              : {best['score']:.5f}")
print(f"Best scale            : {best['scale']:.5f}")
print(f"Uniqueness gap        : {uniqueness_gap:.5f}")
print(f"Recovery distance     : {recovery_distance:.5f}")
print(f"Coord stability std   : ({coord_std[0]:.5f}, {coord_std[1]:.5f})")
print(f"NCC MC std            : {score_std:.5f}")
print("─" * 58)
print("Saved:")
print("  merged_basin_validation_pipeline.png / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

for _obj in [basin, basin_norm, ncc_base, peak_window, mc_coords, mc_scores]:
    del _obj
_ = gc.collect()
print("\nPipeline complete.")

"""
rrg_closure.py
──────────────────────────────────────────────────────────────────────────────
Closure operator: observational equivalence collapse.

─── The missing layer ───────────────────────────────────────────────────────

The pipeline currently does:
  1. generate explanations    → multiscale NCC scan (30 scales)
  2. evaluate against shadow  → scores_log [(scale, NCC_score)]

What was missing:
  3. equivalence collapse     → "does a new scale produce a distinguishable
                                  observational outcome, or is it a
                                  reparameterization of the same match?"

The closure operator answers: given the full trajectory of NCC scores across
scales, at which scale is the instrument maximally sensitive to the boundary
between equivalence classes?  That is the detection event.

─── Mathematical structure ───────────────────────────────────────────────────

Let S = {s_1, s_2, ..., s_k} be the ordered set of scales tried so far.
Let f(s) = NCC surface at scale s  →  a distribution over basin positions.

Two scales s_i, s_j are observationally equivalent if:
    argmax f(s_i) == argmax f(s_j)  [same position class]
    AND  |f(s_i).max() - f(s_j).max()| < ε_score  [indistinguishable score]

The explanation space partitions into equivalence classes under this relation.

Previous closure condition (resolution / "the hand"):
    ΔH(partition after s_k) < epsilon_entropy
    → fires when the partition stops changing; instrument is deep in a basin;
      past J; can no longer detect the boundary.

Current closure condition (susceptibility / "the echo-trail"):
    χ_H(k) = Var(H(partition)[k-chi_window : k+1])
    → detection event = argmax χ_H; instrument is at intermediate rank;
      maximally sensitive to J; never fully resolves.

The RRG pair is genuine when:
  - The susceptibility peak χ_H_max has been reached
  - The dominant class at that peak is internally stable
  - The instrument is at maximum sensitivity, not maximum resolution

─── Integration point ────────────────────────────────────────────────────────

Drop this after the multiscale NCC loop in Celda 3 and in
merged_basin_validation_pipeline.py, before the visualization block.

Input:  scores_log  [(scale, score)]  — already computed
        ncc_surfaces  {scale: ncc_array}  — add one line to the NCC loop
        ppu           pixels-per-unit from basin config

Output: ClosureResult dataclass with all closure metrics + closure_reached bool
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS — result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClosureResult:
    """
    All closure metrics computed from a multiscale NCC run.

    Attributes
    ----------
    closure_reached : bool
        True when the susceptibility peak χ_H has been found.
        The instrument is at maximum sensitivity to J — NOT at resolution.
        "The echo-trail, not the hand."

    closure_scale : float | None
        The scale at which χ_H was maximal.
        This is the detection event: the boundary is most visible here.
        None if fewer than min_scales_before_closure scales were scanned.

    chi_H_curve : np.ndarray  (n_scales,)
        Rolling variance of partition_entropy_curve at each scale k:
            χ_H(k) = Var(H[max(0, k-chi_window) : k+1])
        The peak of this curve is the detection event.

    chi_H_max : float
        Maximum value of chi_H_curve — the susceptibility peak.

    chi_window : int
        Window size used to compute chi_H_curve.

    n_classes : int
        Number of distinct equivalence classes found across all scales.
        A well-defined relation should have n_classes << len(scales).

    dominant_class : int
        Index of the equivalence class that contains the global NCC maximum.

    class_sizes : np.ndarray  (n_classes,)
        Number of scales that map to each equivalence class.

    partition_entropy_curve : np.ndarray  (n_scales,)
        H(partition) after each scale is added.
        Should plateau when the system closes.

    entropy_delta_curve : np.ndarray  (n_scales - 1,)
        |ΔH| between consecutive scales.
        Closure is detected when this drops below epsilon_entropy.

    class_score_mean : np.ndarray  (n_classes,)
        Mean NCC score within each equivalence class.

    class_score_std : np.ndarray  (n_classes,)
        Score spread within each class.
        Low std = the class is internally coherent.

    dominant_class_stability : float
        std of NCC scores within the dominant class.
        Near zero = the match is stable across equivalent scales.

    genuine_pair_certificate : bool
        True when:
          closure_reached (χ_H peak found)
          AND dominant_class_stability < threshold_stability
          AND dominant class is the largest class
        Meaning: "the instrument is at maximum sensitivity to J."
        NOT "the instrument has resolved."  The pair is genuine because
        the boundary is maximally detectable here, not because it has
        been crossed.

    summary : str
        Human-readable summary for paper reporting.
    """
    closure_reached             : bool
    closure_scale               : Optional[float]
    n_classes                   : int
    dominant_class              : int
    class_sizes                 : np.ndarray
    partition_entropy_curve     : np.ndarray
    entropy_delta_curve         : np.ndarray
    chi_H_curve                 : np.ndarray   # rolling susceptibility
    chi_H_max                   : float        # peak value
    chi_window                  : int          # window used
    class_score_mean            : np.ndarray
    class_score_std             : np.ndarray
    dominant_class_stability    : float
    genuine_pair_certificate    : bool
    summary                     : str = field(default="", repr=False)


# ══════════════════════════════════════════════════════════════════════════════
# CORE — equivalence partition
# ══════════════════════════════════════════════════════════════════════════════

def _build_equivalence_partition(
    positions: np.ndarray,       # (n_scales, 2)  [row, col] of argmax per scale
    scores: np.ndarray,          # (n_scales,)    NCC max per scale
    epsilon_px: float = 5.0,     # spatial tolerance in pixels
    epsilon_score: float = 0.02  # score tolerance
) -> np.ndarray:                 # (n_scales,)  class label per scale
    """
    Assigns each scale to an equivalence class.

    Two scales are equivalent if their argmax positions are within
    epsilon_px pixels AND their scores differ by less than epsilon_score.

    This is a greedy single-pass assignment — first scale seen defines
    the class center.  Good enough for 30 scales; not meant for 10k.

    epsilon_px    [⚠ tune] too small → many singleton classes (under-collapse)
                            too large → merges genuinely different matches (over-collapse)
                  Default 5 px is ~0.025 units for BASIN_RES=800, range=4.0
    epsilon_score [⚠ tune] 0.02 is ~2% NCC difference
    """
    n = len(scores)
    labels    = np.full(n, -1, dtype=np.int32)
    centers_p = []   # list of (row, col) class centers
    centers_s = []   # list of score class centers

    next_label = 0

    for i in range(n):
        assigned = False
        for c, (cp, cs) in enumerate(zip(centers_p, centers_s)):
            dist  = np.linalg.norm(positions[i] - cp)
            sdiff = abs(scores[i] - cs)
            if dist <= epsilon_px and sdiff <= epsilon_score:
                labels[i] = c
                assigned   = True
                break
        if not assigned:
            labels[i] = next_label
            centers_p.append(positions[i].copy())
            centers_s.append(scores[i])
            next_label += 1

    return labels


def _partition_entropy(labels: np.ndarray) -> float:
    """Shannon entropy of the class-size distribution."""
    n = len(labels)
    if n == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / n
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — closure_operator
# ══════════════════════════════════════════════════════════════════════════════

def closure_operator(
    scores_log: list[tuple[float, float]],
    ncc_surfaces: dict[float, np.ndarray],
    ppu: float,
    epsilon_px: float = 5.0,
    epsilon_score: float = 0.02,
    chi_window: int = 5,
    threshold_stability: float = 0.015,
    min_scales_before_closure: int = 5,
    verbose: bool = True,
) -> ClosureResult:
    """
    Computes the closure operator over the multiscale NCC explanation space.

    Detection event = peak of susceptibility χ_H, not entropy saturation.
    The instrument stays at intermediate rank — maximally sensitive to J.

    Parameters
    ----------
    scores_log : list[(scale, ncc_score)]
        Output of the multiscale NCC loop.

    ncc_surfaces : dict{scale: np.ndarray}
        Full NCC surface per scale.

    ppu : float
        Pixels per unit — from  ppu = BASIN_RES / (XMAX - XMIN)

    epsilon_px : float
        Spatial tolerance for equivalence (pixels).
        [⚠ tune] Default 5 px ≈ 0.025 units at res=800.

    epsilon_score : float
        Score tolerance for equivalence.
        [⚠ tune] 0.02 works for NCC in [0.5, 0.9].

    chi_window : int
        Rolling window for χ_H = Var(H[k-chi_window : k+1]).
        [⚠ tune] Larger window → smoother susceptibility curve, later peak.
                 Smaller window → more reactive, earlier peak.
                 Default 5 = half of typical 10-scale scans.
                 Analogous to W in the temporal entropy: finite trail depth.

    threshold_stability : float
        Max allowed std of NCC scores within the dominant class.
        [⚠ tune] 0.015 ≈ 1.5% NCC variation.

    min_scales_before_closure : int
        Minimum scales before detection can fire.

    verbose : bool

    Returns
    -------
    ClosureResult
    """
    scales = np.array([s for s, _ in scores_log])
    scores = np.array([sc for _, sc in scores_log])
    n      = len(scales)

    if n == 0:
        raise ValueError("scores_log is empty.")

    # ── Extract argmax positions ───────────────────────────────────────────
    positions = np.zeros((n, 2), dtype=np.float32)
    for i, (s, _) in enumerate(scores_log):
        surf = ncc_surfaces[s]
        idx  = np.unravel_index(np.argmax(surf), surf.shape)
        positions[i] = idx   # (row, col) in pixels

    # ── Build partition incrementally, track entropy curve ────────────────
    partition_entropy_curve = np.zeros(n)
    labels_running          = np.full(n, -1, dtype=np.int32)

    for k in range(1, n + 1):
        labels_k = _build_equivalence_partition(
            positions[:k], scores[:k],
            epsilon_px=epsilon_px,
            epsilon_score=epsilon_score
        )
        labels_running[:k]       = labels_k
        partition_entropy_curve[k - 1] = _partition_entropy(labels_k)

    # ── Entropy delta curve ────────────────────────────────────────────────
    entropy_delta_curve = np.abs(np.diff(partition_entropy_curve))

    # ── Susceptibility curve  χ_H(k) = Var(H[k-chi_window : k+1]) ────────
    # This is the echo-trail mechanism: instead of stopping when ΔH drops,
    # we find where the partition is most volatile — maximum sensitivity to J.
    chi_H_curve = np.zeros(n)
    for k in range(n):
        window_slice     = partition_entropy_curve[max(0, k - chi_window) : k + 1]
        chi_H_curve[k]   = float(np.var(window_slice))

    # Detection event = peak of χ_H after min_scales guard
    chi_H_max     = float(chi_H_curve[min_scales_before_closure:].max())
    peak_idx      = int(np.argmax(chi_H_curve[min_scales_before_closure:]))
    peak_idx     += min_scales_before_closure   # absolute index

    # Boundary peak guard: if the peak lands on the last 2 scales the scan
    # range is too narrow — susceptibility never turned over inside the window.
    # The genuine certificate will be forced False in this case regardless of
    # other conditions, since the detection event is not trustworthy.
    boundary_peak = (peak_idx >= n - 2)
    if boundary_peak and verbose:
        print(f"  [⚠] χ_H peak at boundary (idx={peak_idx}/{n-1}) — "
              f"result unreliable; extend SCALES upper bound or add more scales.")

    closure_reached = True   # always reached — the peak always exists
    closure_scale   = float(scales[peak_idx])

    # ── Final partition (all scales) ──────────────────────────────────────
    final_labels = _build_equivalence_partition(
        positions, scores,
        epsilon_px=epsilon_px,
        epsilon_score=epsilon_score
    )

    unique_classes, class_counts = np.unique(final_labels, return_counts=True)
    n_classes = len(unique_classes)

    # ── Class statistics ──────────────────────────────────────────────────
    class_score_mean = np.zeros(n_classes)
    class_score_std  = np.zeros(n_classes)

    for ci, c in enumerate(unique_classes):
        mask = final_labels == c
        class_score_mean[ci] = scores[mask].mean()
        class_score_std[ci]  = scores[mask].std()

    # ── Dominant class = class containing the global NCC maximum ─────────
    global_best_idx  = int(np.argmax(scores))
    dominant_class   = int(final_labels[global_best_idx])
    dominant_ci      = int(np.where(unique_classes == dominant_class)[0][0])
    dom_stability    = float(class_score_std[dominant_ci])

    # Largest class by count
    largest_class    = int(unique_classes[np.argmax(class_counts)])

    # ── Genuine pair certificate ──────────────────────────────────────────
    # Fires at maximum sensitivity, not maximum resolution.
    # Fix 3: relax strict dominant==largest equality.
    # Accept dominant class if its mean score is within epsilon_score of
    # the largest class — handles phase-boundary cases where the highest-NCC
    # cluster is small but genuinely better than the most-populated one.
    #
    # Additional guards added after false-positive analysis:
    #   - dominant class must have >= MIN_DOMINANT_SIZE members (prevents
    #     a 2-scale singleton from winning via score_gap relaxation)
    #   - boundary_peak must be False (peak at scan edge = unreliable detection)
    MIN_DOMINANT_SIZE = 3
    largest_ci          = int(np.argmax(class_counts))
    largest_class_score = float(class_score_mean[largest_ci])
    dominant_score      = float(class_score_mean[dominant_ci])
    score_gap           = abs(dominant_score - largest_class_score)
    dom_size            = int(class_counts[dominant_ci])

    genuine = (
        closure_reached
        and not boundary_peak
        and dom_stability < threshold_stability
        and dom_size >= MIN_DOMINANT_SIZE
        and (dominant_class == largest_class or score_gap <= epsilon_score)
    )

    # ── Summary ───────────────────────────────────────────────────────────
    lines = [
        "─" * 60,
        "CLOSURE OPERATOR — Susceptibility / Echo-trail",
        "─" * 60,
        f"Scales evaluated          : {n}",
        f"Equivalence classes found : {n_classes}",
        f"Class sizes               : {class_counts.tolist()}",
        f"Dominant class            : {dominant_class}  "
        f"(mean NCC={class_score_mean[dominant_ci]:.4f}, "
        f"σ={dom_stability:.4f})",
        f"χ_H peak scale            : {closure_scale:.4f}  (idx={peak_idx})",
        f"χ_H max                   : {chi_H_max:.6f}",
        f"χ_H window                : {chi_window}",
        f"Final partition entropy   : {partition_entropy_curve[-1]:.5f} nats",
        "─" * 60,
        f"Genuine pair certificate  : {'✓  TRUE' if genuine else '✗  FALSE'}",
        "  Meaning: instrument at maximum sensitivity to J",
        f"    χ_H peak found        : {closure_reached}",
        f"    boundary peak         : {boundary_peak}  →  must be False",
        f"    dominant stability    : σ={dom_stability:.4f} "
        f"< {threshold_stability}  →  {dom_stability < threshold_stability}",
        f"    dominant class size   : {dom_size} >= {MIN_DOMINANT_SIZE}  →  {dom_size >= MIN_DOMINANT_SIZE}",
        f"    dominant == largest   : {dominant_class == largest_class}  "
        f"(score_gap={score_gap:.5f} ≤ ε={epsilon_score}  →  "
        f"{dominant_class == largest_class or score_gap <= epsilon_score})",
        "─" * 60,
    ]
    summary = "\n".join(lines)

    if verbose:
        print(summary)

    return ClosureResult(
        closure_reached          = closure_reached,
        closure_scale            = closure_scale,
        n_classes                = n_classes,
        dominant_class           = dominant_class,
        class_sizes              = class_counts,
        partition_entropy_curve  = partition_entropy_curve,
        entropy_delta_curve      = entropy_delta_curve,
        chi_H_curve              = chi_H_curve,
        chi_H_max                = chi_H_max,
        chi_window               = chi_window,
        class_score_mean         = class_score_mean,
        class_score_std          = class_score_std,
        dominant_class_stability = dom_stability,
        genuine_pair_certificate = genuine,
        summary                  = summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION — panel para incluir en el report existente
# ══════════════════════════════════════════════════════════════════════════════

def plot_closure_panel(
    closure: ClosureResult,
    scores_log: list[tuple[float, float]],
    ax_entropy=None,
    ax_classes=None,
    savepath: str | None = None,
    savekw: dict | None = None,
):
    """
    Genera dos paneles de clausura para incluir en el report de 6 paneles.

    ax_entropy : matplotlib Axes para la curva de entropía de partición
    ax_classes : matplotlib Axes para la distribución de clases

    Si ambos son None crea una figura nueva de 2 paneles.

    Uso en merged_basin_validation_pipeline.py:
        # reemplazar Panel 5 (NCC vs scale) por los dos paneles de clausura
        # o añadir una segunda fila de paneles
    """
    import matplotlib.pyplot as plt

    standalone = (ax_entropy is None and ax_classes is None)
    if standalone:
        fig, (ax_entropy, ax_classes) = plt.subplots(
            1, 2, figsize=(12, 4), facecolor="white"
        )

    scales = np.array([s for s, _ in scores_log])
    scores = np.array([sc for _, sc in scores_log])

    # ── Panel A: partition entropy curve ─────────────────────────────────
    ax_entropy.plot(scales, closure.partition_entropy_curve,
                    "o-", ms=4, lw=1.5, color="#3B82F6", label="H(partition)")

    if closure.closure_scale is not None:
        ax_entropy.axvline(closure.closure_scale, color="red", ls="--", lw=1.2,
                           label=f"closure @ s={closure.closure_scale:.3f}")

    ax_entropy.set_xlabel("scale")
    ax_entropy.set_ylabel("partition entropy (nats)")
    ax_entropy.set_title(
        f"Equivalence closure\n"
        f"{'✓ CLOSED' if closure.closure_reached else '✗ open'}  "
        f"| classes={closure.n_classes}"
    )
    ax_entropy.legend(fontsize=8)

    # ── Panel B: class score distribution ────────────────────────────────
    x = np.arange(closure.n_classes)
    colors = ["#EF4444" if i == closure.dominant_class else "#6B7280"
              for i in range(closure.n_classes)]

    ax_classes.bar(x, closure.class_score_mean, yerr=closure.class_score_std,
                   color=colors, alpha=0.85, capsize=4, width=0.6)
    ax_classes.set_xticks(x)
    ax_classes.set_xticklabels(
        [f"C{i}\n(n={closure.class_sizes[i]})" for i in range(closure.n_classes)],
        fontsize=8
    )
    ax_classes.set_ylabel("mean NCC score")
    ax_classes.set_title(
        f"Class score distribution\n"
        f"genuine={'✓' if closure.genuine_pair_certificate else '✗'}  "
        f"dom_σ={closure.dominant_class_stability:.4f}"
    )

    if standalone:
        plt.tight_layout()
        if savepath:
            kw = savekw or {"dpi": 220, "bbox_inches": "tight"}
            plt.savefig(savepath, **kw)
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION PATCH — exact lines to add to the NCC loop
# ══════════════════════════════════════════════════════════════════════════════

INTEGRATION_PATCH = '''
# ── In merged_basin_validation_pipeline.py ────────────────────────────────
# STEP 1: add ncc_surfaces dict before the NCC loop

ncc_surfaces = {}                      # ← add this line

for s in SCALES:
    result = compute_ncc_surface(basin_norm, phase_recovery, s)
    if result is None:
        continue
    score = result["score"]
    scores_log.append((s, score))
    ncc_surfaces[s] = result["ncc"]    # ← add this line
    ...

# STEP 2: after the loop, run the closure operator

from rrg_closure import closure_operator, plot_closure_panel

closure = closure_operator(
    scores_log   = scores_log,
    ncc_surfaces = ncc_surfaces,
    ppu          = ppu,
    epsilon_px         = 5.0,   # [⚠ tune] spatial tolerance in pixels
    epsilon_score      = 0.02,  # [⚠ tune] NCC score tolerance
    epsilon_entropy    = 0.005, # [⚠ tune] ΔH closure threshold
    threshold_stability= 0.015, # [⚠ tune] max σ for genuine certificate
)

# STEP 3: save to checkpoint
ckpt.save_npz("closure_result",
    closure_reached          = np.bool_(closure.closure_reached),
    closure_scale            = np.float32(closure.closure_scale or -1),
    n_classes                = np.int32(closure.n_classes),
    dominant_class           = np.int32(closure.dominant_class),
    class_sizes              = closure.class_sizes,
    partition_entropy_curve  = closure.partition_entropy_curve,
    entropy_delta_curve      = closure.entropy_delta_curve,
    class_score_mean         = closure.class_score_mean,
    class_score_std          = closure.class_score_std,
    dominant_class_stability = np.float32(closure.dominant_class_stability),
    genuine_pair_certificate = np.bool_(closure.genuine_pair_certificate),
)

# STEP 4: add to visualization report (replaces or extends Panel 5)
plot_closure_panel(closure, scores_log,
                   ax_entropy=ax5,    # pass existing axes or None for new fig
                   ax_classes=ax6)
'''

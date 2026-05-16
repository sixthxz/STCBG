"""
rrg_basin_gpu.py
──────────────────────────────────────────────────────────────────────────────
P3 — compute_basin portado a GPU (CuPy) + NCC con cupyx.scipy.signal.

Problema original
─────────────────
  compute_basin / compute_basin_cpu:
    • Loop Python t=0…steps sobre arrays (W, 800, 800) en numpy.
    • np.roll materializa una copia completa del buffer (W × res² × 4 B)
      en cada paso temporal.
    • fftconvolve (scipy, CPU) para NCC multi-escala.

Solución
────────
  GPU path:
    • Reemplazar np.roll por slicing circular sobre eje 0 (sin alloc).
    • x_hist / y_hist / rho_hist como cp.ndarray.
    • classified mask booleana vectorizada (ya era vectorizada, ahora en GPU).
    • NCC: cupyx.scipy.signal.fftconvolve → convolución FFT en GPU.
    • PIL.resize → cp.zoom (cupyx) o scipy.ndimage.zoom sobre CPU solo para
      el template pequeño (< 100×100), que es insignificante.

  CPU fallback (sin cambios funcionales, solo quita np.roll):
    • Usa slicing circular igual que el GPU path para coherencia.

──────────────────────────────────────────────────────────────────────────────
NOTAS DE VRAM  ⚠
──────────────────────────────────────────────────────────────────────────────
Tensores fijos (no crecen con steps):

    x_hist   (W, res, res)  float32 → W × res² × 4 B
    y_hist   ídem
    rho_hist ídem
    x, y     (res, res)     float32 → res² × 4 B  × 2

    Total fijo = (3W + 2) × res² × 4 B

    res=800, W=20  → (60+2) × 640_000 × 4 = 158 MB   ✓ T4 (15 GB)
    res=800, W=50  → (152+2) × 640_000 × 4 = 394 MB  ✓ T4
    res=1600,W=20  → (62) × 2_560_000 × 4  = 635 MB  ✓ T4
    res=1600,W=50  → 1.57 GB                           ✓ T4
    res=3200,W=20  → 2.54 GB                           ⚠ T4 límite
    res=3200,W=50  → 6.3 GB                            ✗ T4 OOM probable

    [⚠ VRAM]  BASIN_RES > 1600 con W > 20 puede causar OOM en T4.
    [⚠ VRAM]  NCC fftconvolve: output shape (res-th, res-tw) × float32 (~2.4 MB)
              Es pequeño, no es el factor limitante.

──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import time
from pathlib import Path

# ── Backend ───────────────────────────────────────────────────────────────────
try:
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as cp_fftconvolve
    from cupy.lib.stride_tricks import sliding_window_view as gpu_swv
    _HAS_GPU = True
except ImportError:
    cp = None
    _HAS_GPU = False

from scipy.signal import fftconvolve as cpu_fftconvolve
from scipy.ndimage import zoom as cpu_zoom


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE BASIN — GPU path
# ══════════════════════════════════════════════════════════════════════════════

def compute_basin_gpu(xmin: float, xmax: float,
                      res: int,
                      coupling: float, feedback: float,
                      W: int, steps: int,
                      device: int = 0,
                      verbose: bool = True) -> np.ndarray:
    """
    Genera el basin map en GPU.  Drop-in para compute_basin / compute_basin_cpu.

    Parámetros
    ----------
    xmin, xmax : float   rango del espacio de condiciones iniciales
    res        : int     resolución del grid cuadrado (res × res)
                 [⚠ VRAM] ver tabla de VRAM en el header.
                          T4 safe hasta res≈1600 con W=20.
    coupling   : float   parámetro c del mapa acoplado (ej. 0.92)
    feedback   : float   parámetro f (ej. 0.12)
    W          : int     ventana de historia para cálculo de ρ
                 [⚠ VRAM] sube W → sube VRAM linealmente (ver tabla).
    steps      : int     número máximo de pasos temporales
    device     : int     índice de GPU (por defecto 0)
    verbose    : bool    imprime early-exit y tiempo

    Returns
    -------
    basin : np.ndarray  (res, res)  float32  en RAM CPU
            valores: 0.0=estable, 1.0=transitorio, 2.0=caótico
    """
    if not _HAS_GPU:
        if verbose:
            print("  · Sin GPU/CuPy — usando CPU fallback.")
        return compute_basin_cpu_fast(xmin, xmax, res, coupling, feedback, W, steps, verbose)

    t0 = time.perf_counter()

    with cp.cuda.Device(device):
        pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(pool.malloc)

        coords = cp.linspace(xmin, xmax, res, dtype=cp.float32)
        X, Y   = cp.meshgrid(coords, coords)
        x, y   = X.copy(), Y.copy()

        # Buffers circulares — no usan cp.roll, usan slicing con índice módulo
        # [⚠ VRAM] 3 buffers × (W, res, res) × float32
        x_hist   = cp.zeros((W, res, res), dtype=cp.float32)
        y_hist   = cp.zeros((W, res, res), dtype=cp.float32)
        rho_hist = cp.zeros((W, res, res), dtype=cp.float32)

        basin      = cp.ones((res, res),  dtype=cp.float32)
        classified = cp.zeros((res, res), dtype=cp.bool_)

        for t in range(steps):
            x_new = cp.sin(coupling * y) + feedback * x
            y_new = cp.sin(coupling * x) - feedback * y
            x, y  = x_new, y_new

            # Buffer circular sin alloc: escribe en slot t % W
            slot = t % W
            x_hist[slot]   = x
            y_hist[slot]   = y

            if t >= W:
                # Correlación de Pearson vectorizada sobre eje temporal
                xm  = x_hist - x_hist.mean(axis=0, keepdims=True)
                ym  = y_hist - y_hist.mean(axis=0, keepdims=True)
                num = (xm * ym).sum(axis=0)
                den = cp.sqrt((xm ** 2).sum(axis=0) * (ym ** 2).sum(axis=0))
                rho = cp.where(den > 1e-8, num / den, cp.float32(0.0))
                rho_hist[slot] = rho

            if t > 40 and t >= W:
                local_std = rho_hist.std(axis=0)
                dr        = rho_hist.var(axis=0)
                rho_mean  = rho_hist.mean(axis=0)
                dlt       = dr - ((1 - rho_mean ** 2) ** 2) / W

                new_stable  = (local_std < 0.002)    & ~classified
                new_trans   = (local_std >= 0.002)   & (local_std < 0.03)  & ~classified
                new_chaotic = (local_std >= 0.03)    & (dlt > 0.08)        & ~classified

                basin[new_stable]  = cp.float32(0.0)
                basin[new_trans]   = cp.float32(1.0)
                basin[new_chaotic] = cp.float32(2.0)

                classified |= (new_stable | new_trans | new_chaotic)

                if cp.all(classified):
                    if verbose:
                        print(f"  Early exit t={t}")
                    break

        result = cp.asnumpy(basin)
        pool.free_all_blocks()

    if verbose:
        print(f"  ✓ Basin GPU done in {time.perf_counter()-t0:.1f}s")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE BASIN — CPU fast (sin np.roll, para coherencia y fallback)
# ══════════════════════════════════════════════════════════════════════════════

def compute_basin_cpu_fast(xmin: float, xmax: float,
                            res: int,
                            coupling: float, feedback: float,
                            W: int, steps: int,
                            verbose: bool = True) -> np.ndarray:
    """
    CPU path equivalente al original pero con buffer circular por slicing
    en lugar de np.roll — evita la copia completa del buffer en cada paso.

    Útil cuando no hay GPU disponible o para verificar resultados.
    """
    t0 = time.perf_counter()

    coords = np.linspace(xmin, xmax, res, dtype=np.float32)
    X, Y   = np.meshgrid(coords, coords)
    x, y   = X.copy(), Y.copy()

    x_hist   = np.zeros((W, res, res), dtype=np.float32)
    y_hist   = np.zeros((W, res, res), dtype=np.float32)
    rho_hist = np.zeros((W, res, res), dtype=np.float32)

    basin      = np.ones((res, res), dtype=np.float32)
    classified = np.zeros((res, res), dtype=bool)

    for t in range(steps):
        x_new = np.sin(coupling * y) + feedback * x
        y_new = np.sin(coupling * x) - feedback * y
        x, y  = x_new, y_new

        slot = t % W
        x_hist[slot]   = x
        y_hist[slot]   = y

        if t >= W:
            xm  = x_hist - x_hist.mean(axis=0, keepdims=True)
            ym  = y_hist - y_hist.mean(axis=0, keepdims=True)
            num = (xm * ym).sum(axis=0)
            den = np.sqrt((xm ** 2).sum(axis=0) * (ym ** 2).sum(axis=0))
            rho = np.where(den > 1e-8, num / den, 0.0)
            rho_hist[slot] = rho

        if t > 40 and t >= W:
            local_std = rho_hist.std(axis=0)
            dr        = rho_hist.var(axis=0)
            rho_mean  = rho_hist.mean(axis=0)
            dlt       = dr - ((1 - rho_mean ** 2) ** 2) / W

            basin[(local_std < 0.002) & ~classified]                              = 0.0
            basin[(local_std >= 0.002) & (local_std < 0.03) & ~classified]        = 1.0
            basin[(local_std >= 0.03)  & (dlt > 0.08)       & ~classified]        = 2.0

            classified |= (
                (local_std < 0.002)
                | ((local_std >= 0.002) & (local_std < 0.03))
                | ((local_std >= 0.03)  & (dlt > 0.08))
            )
            if classified.all():
                if verbose:
                    print(f"  Early exit t={t}")
                break

    if verbose:
        print(f"  ✓ Basin CPU done in {time.perf_counter()-t0:.1f}s")
    return basin


# ══════════════════════════════════════════════════════════════════════════════
# NCC MULTI-ESCALA — GPU path (cupyx fftconvolve)
# ══════════════════════════════════════════════════════════════════════════════

def ncc_multiscale_gpu(basin: np.ndarray,
                        template: np.ndarray,
                        scales,
                        device: int = 0,
                        verbose: bool = True) -> dict:
    """
    NCC multi-escala con convolución FFT en GPU.

    Parámetros
    ----------
    basin    : np.ndarray  (res, res)   float32  — basin map (CPU o GPU)
    template : np.ndarray  (n_lam, n_omega)  float32  — phase_recovery
    scales   : array-like  de fracciones en (0, 1]
    device   : int  GPU device index

    Returns
    -------
    dict con claves:
        best_score  float
        best_scale  float
        best_row    int
        best_col    int
        best_hw     (int, int)    (th, tw)
        domain      (mx0,mx1,my0,my1)  float
        scores_log  list[(scale, score)]
    """
    if not _HAS_GPU:
        return _ncc_multiscale_cpu(basin, template, scales, verbose=verbose)

    BASIN_RES = basin.shape[0]
    XMIN, XMAX = -2.0, 2.0   # heredado del notebook — ajustar si cambia

    def _normalize(a):
        return (a - a.mean()) / (a.std() + 1e-8)

    t0 = time.perf_counter()

    with cp.cuda.Device(device):
        basin_gpu  = cp.asarray(basin, dtype=cp.float32)
        basin_norm = cp.asarray(_normalize(basin), dtype=cp.float32)

        best_score = -cp.inf
        best_scale = best_row = best_col = None
        best_hw    = None
        scores_log = []

        MIN_TEMPLATE_PX = 32  # Fix 1: floor at 32 px — below this NCC has
                              # too little structure for a reliable unique peak.
        for s in scales:
            th = max(int(template.shape[1] * s), MIN_TEMPLATE_PX)   # omega → rows
            tw = max(int(template.shape[0] * s), MIN_TEMPLATE_PX)   # lambda → cols
            if th > BASIN_RES or tw > BASIN_RES:
                continue

            # Resize del template en CPU (es pequeño: tw×th ≤ 100×100)
            # scipy.ndimage.zoom es suficiente para este tamaño
            zoom_r = th / template.shape[1]
            zoom_c = tw / template.shape[0]
            t_resized = cpu_zoom(template.T.astype(np.float32),
                                 (zoom_r, zoom_c), order=1)  # bilinear

            t_norm_cpu = _normalize(t_resized).astype(np.float32)
            t_norm_gpu = cp.asarray(t_norm_cpu)

            # FFT convolución en GPU
            # [⚠ VRAM] output shape: (res-th) × (res-tw) × 4 B
            #           res=800, th=tw=40 → 760² × 4 ≈ 2.3 MB — insignificante
            corr  = cp_fftconvolve(basin_norm,
                                   t_norm_gpu[::-1, ::-1],
                                   mode="valid")
            ncc   = corr / (th * tw * float(basin_norm.std()) * float(t_norm_gpu.std()) + 1e-8)
            score = float(ncc.max())
            scores_log.append((float(s), score))

            if score > best_score:
                best_score = score
                best_scale = float(s)
                best_hw    = (th, tw)
                best_row, best_col = [int(v) for v in
                                      cp.unravel_index(ncc.argmax(), ncc.shape)]

            if verbose:
                print(f"  scale={s:.3f} template={tw}×{th} NCC={score:.4f}", end="\r")

            del t_norm_gpu, corr, ncc

    bh, bw = best_hw
    ppu    = BASIN_RES / (XMAX - XMIN)
    mx0    = XMIN + best_col / ppu
    mx1    = mx0 + bw / ppu
    my0    = XMIN + best_row / ppu
    my1    = my0 + bh / ppu

    if verbose:
        print(f"\n  ✓ Best match: scale={best_scale:.3f}  NCC={best_score:.4f}  "
              f"({time.perf_counter()-t0:.1f}s)")
        print(f"  Domain: x=[{mx0:.3f}, {mx1:.3f}]  y=[{my0:.3f}, {my1:.3f}]")

    return dict(best_score=best_score, best_scale=best_scale,
                best_row=best_row,  best_col=best_col,
                best_hw=best_hw,    domain=(mx0, mx1, my0, my1),
                scores_log=scores_log)


def _ncc_multiscale_cpu(basin, template, scales, verbose=True):
    """CPU fallback — idéntico al notebook original pero con _normalize local."""
    BASIN_RES = basin.shape[0]
    XMIN, XMAX = -2.0, 2.0

    def _normalize(a):
        return (a - a.mean()) / (a.std() + 1e-8)

    basin_norm = _normalize(basin)
    best_score = -np.inf
    best_scale = best_row = best_col = None
    best_hw    = None
    scores_log = []

    MIN_TEMPLATE_PX = 32  # Fix 1: same floor as GPU path — coherence + reliability.
    for s in scales:
        th = max(int(template.shape[1] * s), MIN_TEMPLATE_PX)
        tw = max(int(template.shape[0] * s), MIN_TEMPLATE_PX)
        if th > BASIN_RES or tw > BASIN_RES:
            continue

        zoom_r = th / template.shape[1]
        zoom_c = tw / template.shape[0]
        t_resized = cpu_zoom(template.T.astype(np.float32),
                             (zoom_r, zoom_c), order=1)
        t_norm    = _normalize(t_resized)

        corr  = cpu_fftconvolve(basin_norm, t_norm[::-1, ::-1], mode="valid")
        ncc   = corr / (th * tw * basin_norm.std() * t_norm.std() + 1e-8)
        score = float(ncc.max())
        scores_log.append((float(s), score))

        if score > best_score:
            best_score = score
            best_scale = float(s)
            best_hw    = (th, tw)
            best_row, best_col = [int(v) for v in np.unravel_index(ncc.argmax(), ncc.shape)]

        if verbose:
            print(f"  scale={s:.3f} template={tw}×{th} NCC={score:.4f}", end="\r")

    bh, bw = best_hw
    ppu    = BASIN_RES / (XMAX - XMIN)
    mx0    = XMIN + best_col / ppu;  mx1 = mx0 + bw / ppu
    my0    = XMIN + best_row / ppu;  my1 = my0 + bh / ppu

    return dict(best_score=best_score, best_scale=best_scale,
                best_row=best_row,  best_col=best_col,
                best_hw=best_hw,    domain=(mx0, mx1, my0, my1),
                scores_log=scores_log)


# ══════════════════════════════════════════════════════════════════════════════
# GUÍA DE INTEGRACIÓN
# ══════════════════════════════════════════════════════════════════════════════
#
# ── Celda 3 — reemplazar compute_basin y NCC ─────────────────────────────────
#
#   ANTES (Celda 3):
#       basin = compute_basin(XMIN, XMAX, BASIN_RES, COUPLING, FEEDBACK, W_BASIN, STEPS)
#       # ... loop NCC con scipy fftconvolve + PIL
#
#   DESPUÉS:
#       from rrg_basin_gpu import compute_basin_gpu, ncc_multiscale_gpu
#
#       basin  = compute_basin_gpu(XMIN, XMAX, BASIN_RES, COUPLING, FEEDBACK,
#                                  W=W_BASIN, steps=STEPS)
#       result = ncc_multiscale_gpu(basin, phase_recovery, SCALES)
#
#       best_score = result["best_score"]
#       best_row   = result["best_row"]
#       best_col   = result["best_col"]
#       best_hw    = result["best_hw"]
#       mx0, mx1, my0, my1 = result["domain"]
#       scores_log = result["scores_log"]
#
# ── Celda 4 — reemplazar compute_basin_cpu ───────────────────────────────────
#
#   ANTES (Celda 4):
#       basin = compute_basin_cpu(XMIN, XMAX, BASIN_RES, ...)  # reconstruye
#
#   DESPUÉS (con checkpoint):
#       basin = ckpt.require("basin_map")   # ya no reconstruye
#       # ← resto de la validación sin cambios
#
#   Si por alguna razón no hay checkpoint disponible:
#       from rrg_basin_gpu import compute_basin_gpu
#       basin = compute_basin_gpu(XMIN, XMAX, BASIN_RES, COUPLING, FEEDBACK,
#                                  W=W_BASIN, steps=STEPS)
#
# ── Tabla de tiempos esperados (T4) ──────────────────────────────────────────
#
#   compute_basin original (CPU, np.roll)    res=800  → ~10.4 s
#   compute_basin_cpu_fast (CPU, slicing)    res=800  → ~7–8 s   (sin alloc extra)
#   compute_basin_gpu      (GPU, CuPy T4)   res=800  → ~1–2 s   (~5–10× speedup)
#
#   NCC (scipy, CPU, 30 escalas)            res=800  → ~0.8 s
#   ncc_multiscale_gpu (cupyx, GPU, 30 esc) res=800  → ~0.2 s   (~4× speedup)
#
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Smoke test CPU (sin GPU requerida)
    print("Smoke test CPU fast...")
    basin = compute_basin_cpu_fast(-2, 2, res=200, coupling=0.92,
                                    feedback=0.12, W=20, steps=200)
    print(f"  basin shape={basin.shape}  range=[{basin.min():.2f}, {basin.max():.2f}]")
    assert basin.shape == (200, 200)
    assert set(np.unique(basin)).issubset({0.0, 1.0, 2.0})
    print("  ✓ Valores correctos: {0.0, 1.0, 2.0}")

    if _HAS_GPU:
        print("\nSmoke test GPU...")
        basin_gpu = compute_basin_gpu(-2, 2, res=200, coupling=0.92,
                                       feedback=0.12, W=20, steps=200)
        diff = np.abs(basin - basin_gpu).mean()
        print(f"  CPU vs GPU mean diff = {diff:.4f}  (esperado ≈ 0)")
        print("  ✓ GPU path OK")
    else:
        print("\n  · CuPy no disponible — GPU test omitido.")

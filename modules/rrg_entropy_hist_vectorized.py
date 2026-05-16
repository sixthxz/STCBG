"""
rrg_entropy_hist_vectorized.py
──────────────────────────────────────────────────────────────────────────────
Reemplazo drop-in del loop  `for b in range(BINS)`  en compute_entropy_streaming
(Celda 2) y compute_entropy (Celda 3) del notebook basin_map_track.

Estrategia: reemplazar el histograma Python-loop por
  GPU  → cp.zeros + scatter_add (atomic, O(n_pairs) por ventana)
  CPU  → np.add.at  o  scipy.ndimage.sum  (equivalente numpy)

Ambas rutas procesan el tensor completo  (M, chunk_t, n_pairs)  de una sola
pasada, sin iterar sobre bins.

──────────────────────────────────────────────────────────────────────────────
NOTAS DE VRAM  ⚠
──────────────────────────────────────────────────────────────────────────────
Los parámetros que pueden causar OOM se marcan con  [⚠ VRAM].
La regla de dedo para el peak de VRAM por chunk:

    peak_bytes ≈ M_BATCH × TIME_CHUNK × (N² × 4  +  n_pairs × 4  +  BINS × 4)
               + sliding_window_view overhead  ≈  M_BATCH × Tw × N × W × 2  (float16)

Para N=50, BINS=48, W=50, los tensores dominantes son:

    Sigma  (M, chunk_t, N, N)   → M×chunk_t × 50² × 4 B
    rho    (M, chunk_t, n_pairs) → M×chunk_t × 1225 × 4 B
    idx    (M, chunk_t, n_pairs) → misma forma, int32 = × 4 B
    hist   (M, chunk_t, BINS)   → M×chunk_t × 48 × 4 B  ← insignificante

Con M_BATCH=10, TIME_CHUNK=128:
    Sigma  → 10 × 128 × 2500 × 4 = 12.8 MB   ✓
    rho    → 10 × 128 × 1225 × 4 = 6.3 MB    ✓

Con M_BATCH=10, TIME_CHUNK=512:
    Sigma  → 51.2 MB   — safe en T4 (15 GB)
    rho    → 25.2 MB   — safe

Pero si se sube N o se elimina el chunking:
    N=200, sin chunk → Sigma = M × Tw × 200² × 4 ≈ varios GB  ← OOM seguro

Parámetros críticos marcados abajo con [⚠ VRAM].
──────────────────────────────────────────────────────────────────────────────
Multi-GPU
──────────────────────────────────────────────────────────────────────────────
Si hay más de una GPU disponible, compute_entropy_multigpu distribuye los
M realismos entre dispositivos usando threading.  Cada GPU procesa su slice
de forma completamente independiente (no hay comunicación inter-GPU).

Requisitos: cupy instalado, todas las GPUs visibles vía CUDA_VISIBLE_DEVICES.
"""

import numpy as np
import threading
import time

# ── Backend auto-detect ───────────────────────────────────────────────────────
try:
    import cupy as cp
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    try:
        cp.cuda.set_reduction_accelerators(["cub"])
    except Exception:
        pass
    from cupy.lib.stride_tricks import sliding_window_view as gpu_swv
    N_GPUS  = cp.cuda.runtime.getDeviceCount()
    BACKEND = f"GPU (CuPy) · {N_GPUS} device(s)"
except ImportError:
    cp      = None
    N_GPUS  = 0
    BACKEND = "CPU (NumPy)"

print(f"Backend: {BACKEND}")


# ══════════════════════════════════════════════════════════════════════════════
# NÚCLEO VECTORIZADO — histograma sin loop Python
# ══════════════════════════════════════════════════════════════════════════════

def _hist_vectorized_gpu(idx, M, chunk_t, BINS):
    """
    Reemplaza:
        hist = cp.zeros((M, chunk_t, BINS))
        for b in range(BINS):
            hist[:, :, b] = cp.sum(idx == b, axis=2)

    Con:
        scatter_add sobre el eje de bins → una sola pasada O(n_pairs).

    idx shape: (M, chunk_t, n_pairs)  dtype int32/int64

    ⚠ VRAM: hist = M × chunk_t × BINS × 4 B  → insignificante.
             El tensor dominante es idx mismo (M × chunk_t × n_pairs × 4 B).
    """
    # Aplanar (M, chunk_t) en una sola dimensión batch para scatter_add
    B   = M * chunk_t                               # batch total
    idx_flat = idx.reshape(B, -1).astype(cp.int32)  # (B, n_pairs)

    hist_flat = cp.zeros((B, BINS), dtype=cp.float32)

    # scatter_add: por cada (b, pair) suma 1 en hist_flat[b, idx_flat[b, pair]]
    # Equivalente a one_hot.sum(axis=1) pero sin materializar el one_hot.
    row_idx = cp.arange(B, dtype=cp.int32)[:, None]          # (B, 1)  broadcast
    cp.scatter_add(hist_flat, (row_idx, idx_flat), cp.ones_like(idx_flat, dtype=cp.float32))

    return hist_flat.reshape(M, chunk_t, BINS)


def _hist_vectorized_cpu(idx, M, chunk_t, BINS):
    """
    Equivalente numpy: np.add.at sobre eje de bins.
    Para arrays grandes es más rápido usar bincount por fila.

    idx shape: (M, chunk_t, n_pairs)
    """
    B        = M * chunk_t
    idx_flat = idx.reshape(B, -1).astype(np.int32)
    hist_flat = np.zeros((B, BINS), dtype=np.float32)

    # bincount vectorizado: cada fila independiente
    for b in range(B):
        hist_flat[b] = np.bincount(idx_flat[b], minlength=BINS).astype(np.float32)

    return hist_flat.reshape(M, chunk_t, BINS)


# ══════════════════════════════════════════════════════════════════════════════
# REEMPLAZO PARA CELDA 2 — compute_entropy_streaming (GPU only, cupy hardcoded)
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy_streaming_v2(X, W,
                                  BINS=48,
                                  TIME_CHUNK=128,   # [⚠ VRAM] sube → más velocidad, más VRAM
                                  EPS=1e-12,
                                  EDGES=None,
                                  IU=None):
    """
    Drop-in para compute_entropy_streaming de Celda 2.

    Cambios vs original:
      • Histograma por scatter_add (sin loop Python sobre bins).
      • EDGES e IU se pueden pasar pre-computados (evitar re-alloc por call).

    Parámetros
    ----------
    X          : cp.ndarray  (M, T, N)  float32
    W          : int         ventana temporal
    BINS       : int         número de bins del histograma de correlaciones
                 [⚠ VRAM]   no sube la VRAM directamente, pero hist = M×chunk_t×BINS×4 B
    TIME_CHUNK : int         tamaño del slab temporal
                 [⚠ VRAM]   peak Sigma ∝ M×TIME_CHUNK×N²×4 B
                             T4 (15 GB): seguro hasta ~512 con M=10, N=50
                             Si N>100 o M>20 → reducir a 64–128
    EPS        : float       estabilidad numérica del log
    EDGES      : cp.ndarray  (BINS+1,)  bordes del histograma en [-1, 1]
                 Pre-computable fuera del scan para evitar malloc por call.
    IU         : tuple       (cp.ndarray, cp.ndarray)  índices upper-triangular de (N,N)
                 Pre-computable fuera del scan.

    Returns
    -------
    H : cp.ndarray  (M, total_windows)
    """
    N = X.shape[2]

    if EDGES is None:
        EDGES = cp.linspace(-1, 1, BINS + 1, dtype=cp.float32)
    if IU is None:
        IU = cp.triu_indices(N, k=1)

    wins         = gpu_swv(X, W, axis=1).transpose(0, 1, 3, 2)  # (M, Tw, W, N)
    total_windows = wins.shape[1]
    H_chunks     = []

    for start in range(0, total_windows, TIME_CHUNK):
        end   = min(start + TIME_CHUNK, total_windows)
        chunk = wins[:, start:end]                  # (M, ct, W, N)
        M_c, ct = chunk.shape[0], chunk.shape[1]

        mu  = chunk.mean(axis=-2, keepdims=True)
        sig = chunk.std(axis=-2,  keepdims=True) + 1e-6
        Z   = (chunk - mu) / sig                    # (M, ct, W, N)

        # Sigma: (M, ct, N, N)
        # [⚠ VRAM] peak aquí: M_c × ct × N² × 4 B
        #          N=50  → 25_000 × 4 = 100 KB por (M,ct)=1
        #          N=200 → 160 KB por (M,ct)=1 → M×ct grande → OOM
        Sigma = cp.einsum(
            'mtiv,mtjv->mtij',
            Z.transpose(0, 1, 3, 2),
            Z.transpose(0, 1, 3, 2)
        ) / W                                        # (M, ct, N, N)

        rho = Sigma[:, :, IU[0], IU[1]]             # (M, ct, n_pairs)

        # [⚠ VRAM] idx: misma forma que rho, int32 (4 B por elemento)
        idx = cp.clip(cp.digitize(rho, EDGES) - 1, 0, BINS - 1).astype(cp.int32)

        # ── HISTOGRAMA VECTORIZADO (sin loop Python) ──────────────────────────
        hist = _hist_vectorized_gpu(idx, M_c, ct, BINS)   # (M, ct, BINS)

        hist /= hist.sum(axis=-1, keepdims=True)
        hist += EPS

        H_chunks.append(-cp.sum(hist * cp.log(hist), axis=-1))  # (M, ct)

        del chunk, mu, sig, Z, Sigma, rho, idx, hist

    return cp.concatenate(H_chunks, axis=1)   # (M, total_windows)


# ══════════════════════════════════════════════════════════════════════════════
# REEMPLAZO PARA CELDA 3 — compute_entropy (backend dual GPU/CPU)
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy_v2(X, W,
                        BINS=48,
                        TIME_CHUNK=128,   # [⚠ VRAM] igual que arriba
                        EPS=1e-12,
                        EDGES=None,
                        IU=None,
                        backend=None):
    """
    Drop-in para compute_entropy de Celda 3.
    Soporta GPU (CuPy) y CPU (NumPy) con histograma vectorizado en ambas rutas.

    backend : str | None
        'GPU' → fuerza CuPy (falla si no está disponible)
        'CPU' → fuerza NumPy
        None  → auto-detect (igual que BACKEND global)
    """
    use_gpu = (cp is not None) if backend is None else (backend == 'GPU')
    xp      = cp if use_gpu else np

    N = X.shape[2]

    if EDGES is None:
        EDGES = xp.linspace(-1, 1, BINS + 1, dtype=xp.float32)
    if IU is None:
        IU = xp.triu_indices(N, k=1)

    if use_gpu:
        wins = gpu_swv(X, W, axis=1).transpose(0, 1, 3, 2)
    else:
        wins = np.lib.stride_tricks.sliding_window_view(X, W, axis=1).transpose(0, 1, 3, 2)

    Tw       = wins.shape[1]
    H_chunks = []

    for start in range(0, Tw, TIME_CHUNK):
        chunk      = wins[:, start:min(start + TIME_CHUNK, Tw)]
        M_c, ct    = chunk.shape[0], chunk.shape[1]

        mu  = chunk.mean(axis=-2, keepdims=True)
        sig = chunk.std(axis=-2,  keepdims=True) + 1e-6
        Z   = (chunk - mu) / sig

        # [⚠ VRAM] Sigma: M_c × ct × N² × 4 B
        Sigma = xp.einsum(
            'mtiv,mtjv->mtij',
            Z.transpose(0, 1, 3, 2),
            Z.transpose(0, 1, 3, 2)
        ) / W

        rho = Sigma[:, :, IU[0], IU[1]]

        # [⚠ VRAM] idx: M_c × ct × n_pairs × 4 B
        idx = xp.clip(xp.digitize(rho, EDGES) - 1, 0, BINS - 1)

        # ── HISTOGRAMA VECTORIZADO ────────────────────────────────────────────
        if use_gpu:
            idx  = idx.astype(cp.int32)
            hist = _hist_vectorized_gpu(idx, M_c, ct, BINS)
        else:
            idx  = idx.astype(np.int32)
            hist = _hist_vectorized_cpu(idx, M_c, ct, BINS)

        hist /= hist.sum(axis=-1, keepdims=True)
        hist += EPS
        H_chunks.append(-(hist * xp.log(hist)).sum(axis=-1))

        del chunk, mu, sig, Z, Sigma, rho, idx, hist

    return xp.concatenate(H_chunks, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-GPU — distribuye M realismos entre N_GPUS dispositivos
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy_multigpu(X_cpu, W,
                              BINS=48,
                              TIME_CHUNK=128,    # [⚠ VRAM] por GPU
                              EPS=1e-12):
    """
    Distribuye los M realismos de X entre todas las GPUs disponibles.
    Cada GPU procesa su slice completo de forma independiente (no AllReduce).

    X_cpu : np.ndarray  (M, T, N)  en RAM CPU
            [⚠ VRAM]   cada GPU recibe M//N_GPUS realismos.
                        Si M es pequeño (e.g. M=10, N_GPUS=4) → 2–3 realismos por GPU.
                        El split no tiene que ser uniforme — el último worker
                        toma el residuo.
            [⚠ VRAM]   cada worker usa TIME_CHUNK para el slab interno.
                        Bajar TIME_CHUNK si VRAM es limitada en alguna GPU.

    Returns
    -------
    H : np.ndarray  (M, total_windows)  en RAM CPU (resultado reunificado)
    """
    if N_GPUS == 0:
        raise RuntimeError("No CuPy / CUDA disponible para multi-GPU.")
    if N_GPUS == 1:
        # Path rápido: una sola GPU, sin threading overhead
        with cp.cuda.Device(0):
            X_gpu = cp.asarray(X_cpu)
            H_gpu = compute_entropy_streaming_v2(X_gpu, W, BINS=BINS,
                                                  TIME_CHUNK=TIME_CHUNK, EPS=EPS)
            return cp.asnumpy(H_gpu)

    M     = X_cpu.shape[0]
    # Dividir M entre GPUs lo más uniforme posible
    sizes  = [M // N_GPUS] * N_GPUS
    sizes[-1] += M % N_GPUS            # residuo va a la última GPU
    starts = [sum(sizes[:i]) for i in range(N_GPUS)]

    results  = [None] * N_GPUS
    errors   = [None] * N_GPUS

    def _worker(device_id, sl_start, sl_size):
        try:
            with cp.cuda.Device(device_id):
                pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(pool.malloc)

                X_sl  = cp.asarray(X_cpu[sl_start:sl_start + sl_size])
                # Pre-computar EDGES e IU en el dispositivo correcto
                N      = X_sl.shape[2]
                EDGES  = cp.linspace(-1, 1, BINS + 1, dtype=cp.float32)
                IU     = cp.triu_indices(N, k=1)

                H_sl  = compute_entropy_streaming_v2(
                    X_sl, W,
                    BINS=BINS, TIME_CHUNK=TIME_CHUNK, EPS=EPS,
                    EDGES=EDGES, IU=IU
                )
                results[device_id] = cp.asnumpy(H_sl)
                pool.free_all_blocks()
        except Exception as e:
            errors[device_id] = e

    threads = []
    for dev, (sl_s, sl_sz) in enumerate(zip(starts, sizes)):
        t = threading.Thread(target=_worker, args=(dev, sl_s, sl_sz), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for dev, err in enumerate(errors):
        if err is not None:
            raise RuntimeError(f"GPU {dev} falló: {err}") from err

    return np.concatenate(results, axis=0)   # (M, total_windows)


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDAD — benchmark comparativo loop vs vectorizado
# ══════════════════════════════════════════════════════════════════════════════

def benchmark(M=10, T=1400, N=50, W=50, BINS=48, TIME_CHUNK=128, n_runs=3):
    """
    Compara el loop original vs scatter_add vectorizado en la GPU disponible.
    Imprime speedup y tiempo mediano.

    [⚠ VRAM] con N=50, M=10, TIME_CHUNK=128 → peak ~20 MB. Safe en cualquier GPU.
              Subir N o TIME_CHUNK para estresar VRAM durante el bench.
    """
    if cp is None:
        print("Sin GPU disponible — benchmark omitido.")
        return

    print(f"\nBenchmark histograma: M={M} T={T} N={N} W={W} BINS={BINS} chunk={TIME_CHUNK}")
    print(f"GPU(s): {N_GPUS}  |  Peak VRAM estimado Sigma: "
          f"{M * TIME_CHUNK * N * N * 4 / 1e6:.1f} MB  [⚠ sube con N y TIME_CHUNK]")

    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((M, T, N)).astype(np.float32)

    # ── ORIGINAL (loop Python) ────────────────────────────────────────────────
    def _original(X):
        from cupy.lib.stride_tricks import sliding_window_view
        _EDGES = cp.linspace(-1, 1, BINS + 1, dtype=cp.float32)
        _IU    = cp.triu_indices(N, k=1)
        wins   = sliding_window_view(X, W, axis=1).transpose(0, 1, 3, 2)
        Tw     = wins.shape[1]
        H_out  = []
        for s in range(0, Tw, TIME_CHUNK):
            chunk = wins[:, s:min(s + TIME_CHUNK, Tw)]
            mu = chunk.mean(axis=-2, keepdims=True)
            sig = chunk.std(axis=-2, keepdims=True) + 1e-6
            Z = (chunk - mu) / sig
            Sigma = cp.einsum('mtiv,mtjv->mtij',
                               Z.transpose(0,1,3,2),
                               Z.transpose(0,1,3,2)) / W
            rho  = Sigma[:, :, _IU[0], _IU[1]]
            idx  = cp.clip(cp.digitize(rho, _EDGES) - 1, 0, BINS - 1)
            M_c, ct = chunk.shape[0], chunk.shape[1]
            hist = cp.zeros((M_c, ct, BINS), dtype=cp.float32)
            for b in range(BINS):          # ← loop Python
                hist[:, :, b] = cp.sum(idx == b, axis=2)
            hist /= hist.sum(axis=-1, keepdims=True)
            hist += 1e-12
            H_out.append(-cp.sum(hist * cp.log(hist), axis=-1))
            del chunk, mu, sig, Z, Sigma, rho, idx, hist
        return cp.concatenate(H_out, axis=1)

    with cp.cuda.Device(0):
        X_gpu = cp.asarray(X_np)

        # Warm-up
        _ = _original(X_gpu); cp.cuda.Device(0).synchronize()
        _ = compute_entropy_streaming_v2(X_gpu, W, BINS=BINS, TIME_CHUNK=TIME_CHUNK)
        cp.cuda.Device(0).synchronize()

        times_orig, times_vec = [], []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _original(X_gpu); cp.cuda.Device(0).synchronize()
            times_orig.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            compute_entropy_streaming_v2(X_gpu, W, BINS=BINS, TIME_CHUNK=TIME_CHUNK)
            cp.cuda.Device(0).synchronize()
            times_vec.append(time.perf_counter() - t0)

    med_o = sorted(times_orig)[n_runs // 2]
    med_v = sorted(times_vec)[n_runs // 2]
    print(f"  Original  (loop Python) : {med_o*1000:.1f} ms")
    print(f"  Vectorizado (scatter_add): {med_v*1000:.1f} ms")
    print(f"  Speedup                 : {med_o/med_v:.2f}×")


# ══════════════════════════════════════════════════════════════════════════════
# GUÍA DE INTEGRACIÓN
# ══════════════════════════════════════════════════════════════════════════════
#
# ── Celda 2 (fig10_relational_phase_diagram_gpu) ─────────────────────────────
#
#   ANTES:
#       H = compute_entropy_streaming(X, W)
#
#   DESPUÉS:
#       from rrg_entropy_hist_vectorized import compute_entropy_streaming_v2
#       H = compute_entropy_streaming_v2(X, W,
#               BINS=BINS, TIME_CHUNK=TIME_CHUNK, EPS=EPS,
#               EDGES=EDGES, IU=IU)   # EDGES e IU ya pre-computados en Celda 2
#
#   Pasar EDGES e IU pre-computados evita re-alocar arrays en cada call del scan.
#
# ── Celda 3 (find_recovery_pattern_in_basin) ─────────────────────────────────
#
#   ANTES:
#       H = compute_entropy(X)      # función local con backend dual
#
#   DESPUÉS:
#       from rrg_entropy_hist_vectorized import compute_entropy_v2
#       H = compute_entropy_v2(X, W,
#               BINS=BINS, TIME_CHUNK=TIME_CHUNK, EPS=EPS,
#               EDGES=EDGES, IU=IU)
#
# ── Multi-GPU (opcional) ─────────────────────────────────────────────────────
#
#   Si tienes ≥2 GPUs y M_TOTAL es grande (e.g. M_TOTAL=80):
#
#       from rrg_entropy_hist_vectorized import compute_entropy_multigpu
#
#       X_cpu = simulate_batch(lam, phi, omega, M=M_TOTAL, seed=seed)
#       if hasattr(X_cpu, 'get'):          # si simulate devuelve CuPy
#           X_cpu = cp.asnumpy(X_cpu)
#       H_cpu = compute_entropy_multigpu(X_cpu, W,
#                   BINS=BINS, TIME_CHUNK=TIME_CHUNK)
#       H     = cp.asarray(H_cpu)         # devolver a GPU para phase_metrics
#
# ── Tabla de VRAM por configuración ──────────────────────────────────────────
#
#   N=50,  M_BATCH=10, TIME_CHUNK=128  → Sigma peak ~12.8 MB  ✓ T4/V100/A100
#   N=50,  M_BATCH=20, TIME_CHUNK=256  → Sigma peak ~51.2 MB  ✓ V100/A100
#   N=100, M_BATCH=10, TIME_CHUNK=128  → Sigma peak ~51.2 MB  ✓ V100/A100
#   N=100, M_BATCH=20, TIME_CHUNK=256  → Sigma peak ~205 MB   ✓ A100, ⚠ T4
#   N=200, M_BATCH=10, TIME_CHUNK=128  → Sigma peak ~205 MB   ✓ A100, ⚠ T4
#   N=200, M_BATCH=10, TIME_CHUNK=512  → Sigma peak ~819 MB   ⚠ solo A100 80 GB
#   N=500, cualquier chunk             → Sigma > 1 GB/item    ✗ OOM probable
#
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    benchmark()

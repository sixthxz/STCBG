"""
rrg_checkpoint.py
──────────────────────────────────────────────────────────────────────────────
P2 — Sistema de checkpoints para eliminar scans duplicados entre celdas.

Problema original
─────────────────
  Celda 2 genera  phase_recovery.npy  (~254 s en T4).
  Celda 3 re-escanea λ×ω completo desde cero (PASO 1) antes de usar el basin.
  Celda 4 reconstruye el basin map completo (~10 s) aunque ya existe en memoria.

Solución
────────
  Guardar: phase_recovery.npy  +  basin_map.npy  +  match_result.npz
  Cargar:  cada celda verifica el checkpoint antes de computar.
  Drive:   helper opcional para montar y sincronizar con Google Drive.

Uso mínimo (Colab)
──────────────────
  # Al inicio de cada celda:
  from rrg_checkpoint import CheckpointStore
  ckpt = CheckpointStore("/content/drive/MyDrive/rrg_checkpoints")

  # Celda 2 — guardar phase_recovery:
  phase_recovery = ckpt.load("phase_recovery") or _compute_phase_recovery(...)
  ckpt.save("phase_recovery", phase_recovery)

  # Celda 3 — cargar sin re-scan:
  phase_recovery = ckpt.require("phase_recovery")   # lanza si no existe
  basin          = ckpt.load("basin_map") or _compute_basin(...)
  ckpt.save("basin_map", basin)

  # Celda 4 — cargar ambos:
  phase_recovery = ckpt.require("phase_recovery")
  basin          = ckpt.require("basin_map")
  match          = ckpt.load_npz("match_result")    # dict con claves score, row, col…
"""

import numpy as np
import os
import json
import time
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# CheckpointStore
# ══════════════════════════════════════════════════════════════════════════════

class CheckpointStore:
    """
    Gestiona guardado / carga de artefactos del pipeline como archivos .npy / .npz.

    Parámetros
    ----------
    root : str | Path
        Directorio base para los checkpoints.
        En Colab con Drive montado usa algo como:
            '/content/drive/MyDrive/rrg_checkpoints'
        Sin Drive usa:
            '/content/rrg_checkpoints'   (se pierde al resetear runtime)
    verbose : bool
        Imprime mensajes de guardado / carga.
    """

    _MANIFEST = "manifest.json"

    def __init__(self, root: str | Path = "/content/rrg_checkpoints",
                 verbose: bool = True):
        self.root    = Path(root)
        self.verbose = verbose
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / self._MANIFEST
        self._manifest = self._load_manifest()

    # ── Manifest (metadata de cada checkpoint) ────────────────────────────────

    def _load_manifest(self) -> dict:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    def _record(self, key: str, path: Path, shape=None, dtype=None):
        self._manifest[key] = {
            "path"      : str(path),
            "saved_at"  : time.strftime("%Y-%m-%dT%H:%M:%S"),
            "shape"     : list(shape) if shape is not None else None,
            "dtype"     : str(dtype) if dtype is not None else None,
        }
        self._save_manifest()

    # ── API principal ─────────────────────────────────────────────────────────

    def save(self, key: str, array: np.ndarray) -> Path:
        """
        Guarda un ndarray como  <root>/<key>.npy
        Devuelve el Path del archivo guardado.
        """
        path = self.root / f"{key}.npy"
        np.save(path, array)
        self._record(key, path, array.shape, array.dtype)
        if self.verbose:
            mb = array.nbytes / 1e6
            print(f"  ✓ ckpt saved  [{key}]  {list(array.shape)}  {array.dtype}  {mb:.2f} MB  → {path}")
        return path

    def load(self, key: str) -> np.ndarray | None:
        """
        Carga el checkpoint si existe, devuelve None si no.
        Usar en patrón:  arr = ckpt.load(key) or compute(...)
        """
        path = self.root / f"{key}.npy"
        if path.exists():
            arr = np.load(path)
            if self.verbose:
                print(f"  ✓ ckpt loaded [{key}]  {list(arr.shape)}  {arr.dtype}  → {path}")
            return arr
        if self.verbose:
            print(f"  · ckpt miss   [{key}]  → will compute")
        return None

    def require(self, key: str) -> np.ndarray:
        """
        Igual que load() pero lanza RuntimeError si no existe.
        Usar en celdas que dependen de pasos anteriores.
        """
        arr = self.load(key)
        if arr is None:
            raise RuntimeError(
                f"Checkpoint '{key}' no encontrado en {self.root}.\n"
                f"Ejecuta la celda que lo genera primero."
            )
        return arr

    # ── NPZ — para resultados compuestos (match, métricas, etc.) ─────────────

    def save_npz(self, key: str, **arrays) -> Path:
        """
        Guarda múltiples arrays como  <root>/<key>.npz
        Ejemplo:
            ckpt.save_npz("match_result",
                          score=np.float32(0.849),
                          row=np.int32(390),
                          col=np.int32(324),
                          scale=np.float32(0.05),
                          domain=np.array([mx0, mx1, my0, my1]))
        """
        path = self.root / f"{key}.npz"
        np.savez_compressed(path, **arrays)
        self._record(key, path)
        if self.verbose:
            keys_str = ", ".join(arrays.keys())
            print(f"  ✓ ckpt saved  [{key}.npz]  keys=[{keys_str}]  → {path}")
        return path

    def load_npz(self, key: str) -> dict | None:
        """
        Carga un .npz como dict  {nombre: ndarray}.
        Devuelve None si no existe.
        """
        path = self.root / f"{key}.npz"
        if path.exists():
            data = dict(np.load(path))
            if self.verbose:
                keys_str = ", ".join(data.keys())
                print(f"  ✓ ckpt loaded [{key}.npz]  keys=[{keys_str}]  → {path}")
            return data
        if self.verbose:
            print(f"  · ckpt miss   [{key}.npz]  → will compute")
        return None

    def require_npz(self, key: str) -> dict:
        data = self.load_npz(key)
        if data is None:
            raise RuntimeError(
                f"Checkpoint '{key}.npz' no encontrado en {self.root}.\n"
                f"Ejecuta la celda que lo genera primero."
            )
        return data

    # ── Utilidades ────────────────────────────────────────────────────────────

    def exists(self, key: str) -> bool:
        return (self.root / f"{key}.npy").exists() or \
               (self.root / f"{key}.npz").exists()

    def delete(self, key: str):
        for ext in (".npy", ".npz"):
            p = self.root / f"{key}{ext}"
            if p.exists():
                p.unlink()
                self._manifest.pop(key, None)
                self._save_manifest()
                if self.verbose:
                    print(f"  ✗ ckpt deleted [{key}]")

    def status(self):
        """Imprime tabla de checkpoints disponibles."""
        print(f"\nCheckpoint store: {self.root}")
        print(f"{'KEY':<25} {'SHAPE':<22} {'DTYPE':<10} {'SAVED AT'}")
        print("─" * 75)
        if not self._manifest:
            print("  (vacío)")
            return
        for key, meta in self._manifest.items():
            shape = str(meta.get("shape", "?"))
            dtype = str(meta.get("dtype", "?"))
            saved = meta.get("saved_at", "?")
            print(f"  {key:<23} {shape:<22} {dtype:<10} {saved}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Google Drive helper
# ══════════════════════════════════════════════════════════════════════════════

def mount_drive_and_get_store(drive_path: str = "MyDrive/rrg_checkpoints",
                               verbose: bool = True) -> "CheckpointStore":
    """
    Monta Google Drive (si no está montado) y devuelve un CheckpointStore
    apuntando a la carpeta indicada.

    Uso en Colab:
        from rrg_checkpoint import mount_drive_and_get_store
        ckpt = mount_drive_and_get_store("MyDrive/rrg_checkpoints")
    """
    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive", force_remount=False)
            if verbose:
                print("  ✓ Google Drive montado.")
        else:
            if verbose:
                print("  · Google Drive ya montado.")
    except ImportError:
        if verbose:
            print("  · No estamos en Colab — usando path local.")

    full_path = Path("/content/drive") / drive_path
    return CheckpointStore(full_path, verbose=verbose)


# ══════════════════════════════════════════════════════════════════════════════
# PLANTILLAS DE INTEGRACIÓN POR CELDA
# ══════════════════════════════════════════════════════════════════════════════

CELL2_TEMPLATE = '''
# ── Inicio Celda 2 ─────────────────────────────────────────────────────────
from rrg_checkpoint import CheckpointStore
ckpt = CheckpointStore("/content/drive/MyDrive/rrg_checkpoints")

phase_recovery = ckpt.load("phase_recovery")

if phase_recovery is None:
    # ← tu scan λ×ω existente aquí (sin cambios)
    phase_recovery = np.zeros((len(LAMBDAS), len(OMEGAS)), dtype=np.float32)
    for i, lam in enumerate(LAMBDAS):
        for j, omega in enumerate(OMEGAS):
            ...  # compute_entropy_streaming_v2(...)
    ckpt.save("phase_recovery", phase_recovery)
else:
    print("  · phase_recovery cargado desde checkpoint — scan omitido (~254 s ahorrados)")
'''

CELL3_TEMPLATE = '''
# ── Inicio Celda 3 ─────────────────────────────────────────────────────────
from rrg_checkpoint import CheckpointStore
ckpt = CheckpointStore("/content/drive/MyDrive/rrg_checkpoints")

# ← PASO 1 ya no re-escanea: carga directamente
phase_recovery = ckpt.require("phase_recovery")

# ← PASO 2: basin map
basin = ckpt.load("basin_map")
if basin is None:
    basin = compute_basin_gpu(...)   # ← ver rrg_basin_gpu.py
    ckpt.save("basin_map", basin)

# ← NCC multi-escala (sin cambios)
...

# Guardar resultado del match para Celda 4
ckpt.save_npz("match_result",
              score=np.float32(best_score),
              scale=np.float32(best_scale),
              row=np.int32(best_row),
              col=np.int32(best_col),
              bh=np.int32(bh),
              bw=np.int32(bw),
              domain=np.array([mx0, mx1, my0, my1], dtype=np.float32))
'''

CELL4_TEMPLATE = '''
# ── Inicio Celda 4 ─────────────────────────────────────────────────────────
from rrg_checkpoint import CheckpointStore
ckpt = CheckpointStore("/content/drive/MyDrive/rrg_checkpoints")

phase_recovery = ckpt.require("phase_recovery")
basin          = ckpt.require("basin_map")        # ← ya no reconstruye (~10 s ahorrados)
match          = ckpt.require_npz("match_result") # score, row, col, domain…

best_score = float(match["score"])
best_row   = int(match["row"])
best_col   = int(match["col"])
bh, bw     = int(match["bh"]), int(match["bw"])
mx0, mx1, my0, my1 = match["domain"]

# ← resto de la validación sin cambios
'''


if __name__ == "__main__":
    # Demo rápido sin GPU
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = CheckpointStore(tmp, verbose=True)

        arr = np.random.rand(40, 40).astype(np.float32)
        ckpt.save("phase_recovery", arr)

        loaded = ckpt.load("phase_recovery")
        assert np.allclose(arr, loaded), "Mismatch!"

        ckpt.save_npz("match_result",
                      score=np.float32(0.849),
                      row=np.int32(390),
                      col=np.int32(324))
        m = ckpt.load_npz("match_result")
        assert float(m["score"]) == 0.849

        ckpt.status()
        print("✓ CheckpointStore OK")

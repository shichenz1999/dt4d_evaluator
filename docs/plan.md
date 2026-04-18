# dt4d_evaluator — Design Plan

## Goal

A standalone Python package that evaluates the quality of fitted mesh sequences
produced by `dt4d_optimizer` against ground-truth sequences from the DT4D HDF5
dataset.

**Inputs**
- Fitted sequence — `(T, V, 3)` float32 `.npy`, output of `dt4d_optimizer`
- Ground-truth sequence — `(T, V, 3)` float32, loaded from `dt4d.hdf5`

**Output**

Per-sequence metrics, aggregated into CSVs:

| Column | Description |
|---|---|
| `animal` | Animal ID (e.g. `bear3EP`) |
| `seq` | Rig source sequence name (e.g. `bear3EP_Idle3`) |
| `target` | Target sequence name (same as `seq` for reconstruction) |
| `l1` | Mean per-vertex L1 distance averaged over all frames |
| `l2` | Mean per-vertex L2 distance averaged over all frames |
| `chamfer_l1` | Mean Chamfer-L1 distance over all frames |
| `chamfer_l2` | Mean Chamfer-L2 distance over all frames |

---

## Repository Layout

```
dt4d_evaluator/
├── docs/
│   └── plan.md          ← this file
├── evaluator/
│   ├── __init__.py      ← public API: l1, l2, chamfer_l1, chamfer_l2
│   └── metrics.py       ← metric implementations (NumPy + PyTorch GPU)
├── run.py               ← single sequence evaluation
├── batch.py             ← batch evaluation over an optimizer output directory
├── input/               ← gitignored
└── output/              ← gitignored, CSV results
```

---

## Module Responsibilities

### `evaluator/metrics.py`

`l1` and `l2` use NumPy (fast, vertex correspondence required).
`chamfer_l1` and `chamfer_l2` use PyTorch with GPU acceleration; pairwise
distances are computed in chunks to avoid OOM on large meshes.

| Function | Signature | Description |
|---|---|---|
| `l1` | `(fitted, target) → (T,)` | Per-frame mean per-vertex L1 distance; requires vertex correspondence |
| `l2` | `(fitted, target) → (T,)` | Per-frame mean per-vertex L2 distance; requires vertex correspondence |
| `chamfer_l1` | `(fitted, target, device) → (T,)` | Per-frame bidirectional Chamfer, mean of sqrt(min d²) |
| `chamfer_l2` | `(fitted, target, device) → (T,)` | Per-frame bidirectional Chamfer, mean of min d² |

### `evaluator/__init__.py`

```python
from .metrics import l1, l2, chamfer_l1, chamfer_l2
__all__ = ['l1', 'l2', 'chamfer_l1', 'chamfer_l2']
```

### `run.py` (single sequence)

Evaluate one fitted sequence against a ground-truth sequence.

```
python run.py --fitted fitted.npy --gt gt.npy
```

Prints summary metrics to stdout.

Arguments: `--fitted`, `--gt`.

### `batch.py` (batch evaluation)

Scans an optimizer `output_dir`, loads ground truth from HDF5, computes metrics
for every fitted sequence, and writes CSVs split by mode.

```
python batch.py \
  --fitted_dir /path/to/dt4d_optimizer/output \
  --hdf5       dt4d.hdf5 \
  --output_dir output/
```

Writes four files:
- `output/reconstruction_results.csv` — per-sequence metrics for reconstruction
- `output/reconstruction_summary.csv` — mean metrics for reconstruction
- `output/transfer_results.csv` — per-sequence metrics for transfer
- `output/transfer_summary.csv` — mean metrics for transfer

- **reconstruction**: ground truth = same sequence as the rig (`animal/seq` from HDF5)
- **transfer**: ground truth = target sequence read from `meta.json` in the same folder
- Either `reconstruction/` or `transfer/` folder may be absent — missing modes are skipped.

Arguments: `--fitted_dir`, `--hdf5`, `--output_dir`.

---

## Metrics

### L1 (Mean Per-Vertex L1 Distance)

Requires vertex correspondence. For each frame, compute the L1 (Manhattan)
distance between each fitted vertex and its corresponding target vertex, then
average over all vertices.

```
l1[t] = mean_v( |fitted[t,v,x] - target[t,v,x]|
              + |fitted[t,v,y] - target[t,v,y]|
              + |fitted[t,v,z] - target[t,v,z]| )
```

### L2 (Mean Per-Vertex L2 Distance)

Requires vertex correspondence. For each frame, compute the Euclidean distance
between each fitted vertex and its corresponding target vertex, then average
over all vertices.

```
l2[t] = mean_v( sqrt( (fitted[t,v,x] - target[t,v,x])²
                    + (fitted[t,v,y] - target[t,v,y])²
                    + (fitted[t,v,z] - target[t,v,z])² ) )
```

### Chamfer-L1

Bidirectional nearest-neighbour Chamfer distance. For each nearest neighbour,
uses Euclidean distance (sqrt). Does not require vertex correspondence.

```
chamfer_l1[t] = ( mean_{v} min_{u} ||fitted[t,v] - target[t,u]||_2
               +  mean_{u} min_{v} ||target[t,u] - fitted[t,v]||_2 ) / 2
```

### Chamfer-L2

Bidirectional nearest-neighbour Chamfer distance using squared Euclidean
distance. Does not require vertex correspondence.

```
chamfer_l2[t] = ( mean_{v} min_{u} ||fitted[t,v] - target[t,u]||²
               +  mean_{u} min_{v} ||target[t,u] - fitted[t,v]||² ) / 2
```

---

## Dependencies

- `torch` — GPU-accelerated Chamfer distance
- `numpy`
- `h5py` — loading ground-truth sequences from `dt4d.hdf5`
- `pandas` — writing results CSV
- `tqdm` — progress bars

# dt4d_evaluator

Evaluates fitted mesh sequences from `dt4d_optimizer` against DT4D ground truth.

## Usage

**Single sequence**
```bash
python run.py --fitted fitted.npy --gt gt.npy
```

**Batch**
```bash
python batch.py \
  --fitted_dir /path/to/dt4d_optimizer/output \
  --hdf5       /path/to/dt4d.hdf5 \
  --output_dir output/
```

## Metrics

| Metric | Description |
|---|---|
| `l1` | Mean per-vertex L1 distance (requires vertex correspondence) |
| `l2` | Mean per-vertex L2 distance (requires vertex correspondence) |
| `chamfer_l1` | Bidirectional Chamfer, mean of sqrt(min d²) |
| `chamfer_l2` | Bidirectional Chamfer, mean of min d² |

## Output

```
output/
├── reconstruction_results.csv   ← per-sequence metrics
├── reconstruction_summary.csv   ← mean over all sequences
├── transfer_results.csv
└── transfer_summary.csv
```

## Dependencies

`torch`, `numpy`, `h5py`, `pandas`, `tqdm`

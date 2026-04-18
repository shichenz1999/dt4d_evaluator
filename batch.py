#!/usr/bin/env python3
"""
Batch evaluation of dt4d_optimizer output against DT4D ground truth.

Scans an optimizer output directory for fitted_vertices.npy files, loads the
corresponding ground-truth sequences from HDF5, and writes a summary CSV.

Output directory structure expected (from dt4d_optimizer):
  fitted_dir/
  ├── reconstruction/
  │   └── <animal>/
  │       └── <seq>/
  │           └── fitted_vertices.npy
  └── transfer/
      └── <animal>/
          ├── fitted_vertices.npy
          └── meta.json

Usage:
  python batch.py \
    --fitted_dir /path/to/dt4d_optimizer/output \
    --hdf5       dt4d.hdf5 \
    --output_dir output/
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluator import l1, l2, chamfer_l1, chamfer_l2


def load_gt(hdf5_path: str, key: str) -> np.ndarray:
    with h5py.File(hdf5_path, 'r') as f:
        return f[key]['vertices'][:].astype(np.float32)


def collect_jobs(fitted_dir: Path) -> list[dict]:
    """
    Walk fitted_dir and return one job dict per fitted_vertices.npy.
    Each dict has: mode, animal, seq, fitted_path, target_seq.
    """
    jobs = []

    # reconstruction: fitted_dir/reconstruction/<animal>/<seq>/fitted_vertices.npy
    recon_dir = fitted_dir / 'reconstruction'
    if recon_dir.exists():
        for p in sorted(recon_dir.glob('**/fitted_vertices.npy')):
            seq    = p.parent.name
            animal = p.parent.parent.name
            jobs.append(dict(mode='reconstruction', animal=animal, seq=seq,
                             fitted_path=p, target_seq=seq))

    # transfer: fitted_dir/transfer/<animal>/fitted_vertices.npy + meta.json
    transfer_dir = fitted_dir / 'transfer'
    if transfer_dir.exists():
        for p in sorted(transfer_dir.glob('*/fitted_vertices.npy')):
            animal    = p.parent.name
            meta_path = p.parent / 'meta.json'
            if not meta_path.exists():
                print(f'  [skip] no meta.json for {animal}')
                continue
            meta      = json.loads(meta_path.read_text())
            jobs.append(dict(mode='transfer', animal=animal, seq=meta['rig'],
                             fitted_path=p, target_seq=meta['target']))

    return jobs


def _evaluate_jobs(jobs: list[dict], hdf5_path: str) -> pd.DataFrame:
    rows = []
    for job in tqdm(jobs, desc='evaluating'):
        fitted = np.load(str(job['fitted_path'])).astype(np.float32)
        gt_key = f"{job['animal']}/{job['target_seq']}"
        gt     = load_gt(hdf5_path, gt_key)

        if fitted.shape != gt.shape:
            print(f"  [skip] shape mismatch: fitted {fitted.shape} vs gt {gt.shape} ({gt_key})")
            continue

        l1_frames    = l1(fitted, gt)
        l2_frames    = l2(fitted, gt)
        cd_l1_frames = chamfer_l1(fitted, gt)
        cd_l2_frames = chamfer_l2(fitted, gt)

        rows.append(dict(
            animal          = job['animal'],
            seq             = job['seq'],
            target          = job['target_seq'],
            l1              = float(l1_frames.mean()),
            l2              = float(l2_frames.mean()),
            chamfer_l1      = float(cd_l1_frames.mean()),
            chamfer_l2      = float(cd_l2_frames.mean()),
        ))
    return pd.DataFrame(rows)


def run_batch(fitted_dir: str, hdf5_path: str, output_dir: str):
    fitted_dir = Path(fitted_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_jobs = collect_jobs(fitted_dir)
    print(f'Found {len(all_jobs)} fitted sequences under {fitted_dir}')

    metric_cols = ['l1', 'l2', 'chamfer_l1', 'chamfer_l2']

    for mode in ('reconstruction', 'transfer'):
        jobs = [j for j in all_jobs if j['mode'] == mode]
        if not jobs:
            continue

        df = _evaluate_jobs(jobs, hdf5_path)

        results_csv = output_dir / f'{mode}_results.csv'
        df.to_csv(str(results_csv), index=False)
        print(f'Saved: {results_csv}  ({len(df)} rows)')

        summary = df[metric_cols].mean().to_frame(name='mean').T
        summary_csv = output_dir / f'{mode}_summary.csv'
        summary.to_csv(str(summary_csv), index=False)
        print(f'Saved: {summary_csv}')
        print(f'[{mode}]')
        print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Batch dt4d fitting evaluation')
    parser.add_argument('--fitted_dir',  required=True, help='Root optimizer output directory')
    parser.add_argument('--hdf5',        required=True, help='Path to dt4d.hdf5')
    parser.add_argument('--output_dir',  required=True, help='Directory to write result CSVs')
    args = parser.parse_args()

    run_batch(
        fitted_dir = args.fitted_dir,
        hdf5_path  = args.hdf5,
        output_dir = args.output_dir,
    )


if __name__ == '__main__':
    main()

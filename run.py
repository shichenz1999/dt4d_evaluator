#!/usr/bin/env python3
"""
Evaluate a single fitted mesh sequence against a ground-truth sequence.

Inputs:
  --fitted  fitted.npy    optimizer output, shape (T, V, 3) float32
  --gt      gt.npy        ground-truth sequence, shape (T, V, 3) float32

Output:
  Per-frame and summary metrics printed to stdout.

Usage:
  python run.py --fitted fitted.npy --gt gt.npy
"""

import argparse

import numpy as np

from evaluator import l1, l2, chamfer_l1, chamfer_l2


def main():
    parser = argparse.ArgumentParser(description='Single-sequence mesh fitting evaluation')
    parser.add_argument('--fitted', required=True, help='Path to fitted sequence .npy, shape (T, V, 3)')
    parser.add_argument('--gt',     required=True, help='Path to ground-truth sequence .npy, shape (T, V, 3)')
    args = parser.parse_args()

    fitted = np.load(args.fitted).astype(np.float32)
    gt     = np.load(args.gt).astype(np.float32)

    assert fitted.shape == gt.shape, \
        f'Shape mismatch: fitted {fitted.shape} vs gt {gt.shape}'

    l1_frames   = l1(fitted, gt)
    l2_frames   = l2(fitted, gt)
    cd_l1_frames = chamfer_l1(fitted, gt)
    cd_l2_frames = chamfer_l2(fitted, gt)

    print(f'Frames    : {fitted.shape[0]}')
    print(f'L1        : {l1_frames.mean():.6f}')
    print(f'L2        : {l2_frames.mean():.6f}')
    print(f'Chamfer-L1: {cd_l1_frames.mean():.6f}')
    print(f'Chamfer-L2: {cd_l2_frames.mean():.6f}')


if __name__ == '__main__':
    main()

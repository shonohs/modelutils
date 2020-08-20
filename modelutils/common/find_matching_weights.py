"""Given two dictionaries for weights, find pairs of matching weights"""

import argparse
import io
import pathlib
import sys
import numpy as np


def find_matching_weights(filepath0, filepath1):
    dict0 = np.load(io.BytesIO(filepath0.read_bytes()), allow_pickle=True).item()
    dict1 = np.load(io.BytesIO(filepath1.read_bytes()), allow_pickle=True).item()

    for key in dict0:
        matched_key, d = min(((k, _distance(dict0[key], v)) for k, v in dict1.items()), key=lambda x: x[1])
        if d < sys.maxsize:
            print(f"{key},{matched_key},{d}")
        else:
            print(f"Failed to find: {key}", file=sys.stderr)


def _distance(value0, value1):
    if value0.size != value1.size:
        return sys.maxsize

    min_max_mean = np.array([np.min(value0) - np.min(value1), np.max(value0) - np.max(value1), np.mean(value0) - np.mean(value1)])
    return np.linalg.norm(min_max_mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npy_filepath0', type=pathlib.Path)
    parser.add_argument('npy_filepath1', type=pathlib.Path)

    args = parser.parse_args()

    if not args.npy_filepath0.exists():
        args.error(f"Input not found: {args.npy_filepath0}")
    if not args.npy_filepath1.exists():
        args.error(f"Input not found: {args.npy_filepath1}")

    find_matching_weights(args.npy_filepath0, args.npy_filepath1)


if __name__ == '__main__':
    main()

import argparse
import numpy as np
from ..common.utils import write_output_npy


def rand(dims):
    zeros = np.random.rand(*dims).astype(np.float32)
    write_output_npy(zeros)


def main():
    parser = argparse.ArgumentParser(description="Create np.random.rand() array")
    parser.add_argument('dims', nargs='+', type=int)

    args = parser.parse_args()
    rand(args.dims)


if __name__ == '__main__':
    main()

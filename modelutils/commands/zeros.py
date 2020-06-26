import argparse
import io
import pathlib
import sys
import numpy as np


def zeros(dims, value):
    zeros = np.zeros(dims, dtype=np.float32)
    if value:
        zeros += value

    buf = io.BytesIO()
    np.save(buf, zeros)
    buf.seek(0)
    sys.stdout.buffer.write(buf.read())


def main():
    parser = argparse.ArgumentParser(description="Create np.zeros() array")
    parser.add_argument('dims', nargs='+', type=int)
    parser.add_argument('--value', '-v', type=float)

    args = parser.parse_args()
    zeros(args.dims, args.value)


if __name__ == '__main__':
    main()

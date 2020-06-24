import argparse
import io
import pathlib
import sys
import numpy as np


def zeros(dims):
    zeros = np.zeros(dims, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, zeros)
    buf.seek(0)
    sys.stdout.buffer.write(buf.read())


def main():
    parser = argparse.ArgumentParser(description="Create np.zeros() array")
    parser.add_argument('dims', nargs='+', type=int)

    args = parser.parse_args()
    zeros(args.dims)


if __name__ == '__main__':
    main()

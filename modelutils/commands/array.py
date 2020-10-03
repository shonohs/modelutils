import argparse
import numpy as np
from ..common.utils import write_output_npy


_STR_TO_DTYPE = {'int32': np.int32,
                 'int8': np.int8,
                 'float32': np.float32}


def array(values, dtype_str, shape):
    dtype = _STR_TO_DTYPE[dtype_str]

    new_array = np.fromstring(' '.join(values), dtype=dtype, sep=' ')
    if shape is not None:
        shape = [int(s) for s in shape]
        new_array = new_array.reshape(shape)
    write_output_npy(new_array)


def main():
    parser = argparse.ArgumentParser(description="Create a numpy array")
    parser.add_argument('values', nargs='*', help="A string containing the data.")
    parser.add_argument('--dtype', default='float32', choices=_STR_TO_DTYPE.keys())
    parser.add_argument('--shape', nargs='*', default=None, help="Shape for the array. If empty, 0-D array will be created.")

    args = parser.parse_args()
    array(args.values, args.dtype, args.shape)

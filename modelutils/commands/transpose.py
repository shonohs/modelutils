import argparse
import pathlib
from ..common.utils import read_input_npy, write_output_npy


def transpose(input_filepath, perms):
    input_data = read_input_npy(input_filepath)
    output_data = input_data.transpose(perms)
    write_output_npy(output_data)


def main():
    parser = argparse.ArgumentParser(description="Get one array from a npy dictionary")
    parser.add_argument('npy_filepath', nargs='?', type=pathlib.Path, help="Filepath to the npy file. If not specified, read from stdin.")
    parser.add_argument('--perm', nargs='+', required=True, type=int)

    args = parser.parse_args()

    transpose(args.npy_filepath, args.perm)

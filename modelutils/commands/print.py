import argparse
import pathlib
from ..common.utils import read_input_npy


def npy_print(input_filepath):
    input_data = read_input_npy(input_filepath)
    print(input_data)


def main():
    parser = argparse.ArgumentParser(description="Get one array from a npy dictionary")
    parser.add_argument('npy_filepath', nargs='?', type=pathlib.Path, help="Filepath to the npy file. If not specified, read from stdin.")

    args = parser.parse_args()
    npy_print(args.npy_filepath)

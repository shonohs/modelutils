import argparse
import pathlib
import sys
import numpy as np
from ..common.utils import read_input_npy


def npy_print(input_filepath, show_stat, show_all):
    input_data = read_input_npy(input_filepath)
    if show_all:
        np.set_printoptions(threshold=sys.maxsize)

    if show_stat:
        print(f"Shape: {input_data.shape} min: {np.min(input_data)} max: {np.max(input_data)} mean: {np.mean(input_data)}")
    else:
        print(input_data)


def main():
    parser = argparse.ArgumentParser(description="Get one array from a npy dictionary")
    parser.add_argument('npy_filepath', nargs='?', type=pathlib.Path, help="Filepath to the npy file. If not specified, read from stdin.")
    parser.add_argument('--stat', '-s', action='store_true')
    parser.add_argument('--all', '-a', action='store_true')

    args = parser.parse_args()
    npy_print(args.npy_filepath, args.stat, args.all)

import argparse
import onnx
import os
import sys


def dump(filename, save_to_file):
    onnx_model = onnx.load(filename)
    if save_to_file:
        save_filename = filename + '.dump'
        if os.path.exists(save_filename):
            raise RuntimeError(f"{save_filename} already exists.")
        with open(save_filename, 'w') as f:
            f.write(str(onnx_model))
    else:
        print(onnx_model)


def main():
    parser = argparse.ArgumentParser('Dump a ONNX model to text')
    parser.add_argument('onnx_filename', type=str, help='Filename for the input onnx file')
    parser.add_argument('-s', '--save', action='store_true', help='Save to .dump file')

    args = parser.parse_args()
    dump(args.onnx_filename, args.save)


if __name__ == '__main__':
    main()

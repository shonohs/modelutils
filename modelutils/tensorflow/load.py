import argparse
import sys
import onnx
import tensorflow
from google.protobuf import text_format


def load(in_filename, out_filename):
    model = tensorflow.compat.v1.GraphDef()
    with open(in_filename, 'r') as f:
        text_format.Parse(f.read(), model)

    with open(out_filename, 'wb') as f:
        f.write(model.SerializeToString())


def main():
    parser = argparse.ArgumentParser('Convert a text file to ONNX model')
    parser.add_argument('text_filename', type=str, help='Filename for the input text file')
    parser.add_argument('model_filename', type=str, help='Output TF file path')

    args = parser.parse_args()
    load(args.text_filename, args.model_filename)


if __name__ == '__main__':
    main()

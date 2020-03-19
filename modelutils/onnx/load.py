import argparse
import onnx
from google.protobuf import text_format


def load(in_filename, out_filename):
    model = onnx.ModelProto()
    with open(in_filename, 'r') as f:
        text_format.Parse(f.read(), model)

    onnx.save(model, out_filename)


def main():
    parser = argparse.ArgumentParser('Load a ONNX model from text')
    parser.add_argument('text_filename', type=str, help='Filename for the input text file')
    parser.add_argument('model_filename', type=str, help='Output ONNX file path')

    args = parser.parse_args()
    load(args.text_filename, args.model_filename)


if __name__ == '__main__':
    main()

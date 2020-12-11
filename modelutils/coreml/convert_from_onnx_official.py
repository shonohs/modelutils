import argparse
import pathlib
import coremltools


def convert(onnx_filepath, output_filepath):
    model = coremltools.converters.onnx.convert(model=str(onnx_filepath))
    model.save(output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    convert(args.onnx_filepath, args.output_filepath)


if __name__ == '__main__':
    main()

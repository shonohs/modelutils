import argparse
import pathlib
from ..caffe.load import load_model as caffe_load_model
from ..onnx.load import load_model as onnx_load_model
from ..tensorflow.load import load_model as tensorflow_load_model


HANDLERS = {'caffe': caffe_load_model,
            'onnx': onnx_load_model,
            'tensorflow': tensorflow_load_model}

KNOWN_SUFFIXES = {'.mlmodel': 'coreml',
                  '.onnx': 'onnx',
                  '.pb': 'tensorflow',
                  '.prototxt': 'caffe'}


def _detect_type_from_suffix(filepath):
    for suffix in KNOWN_SUFFIXES:
        if suffix in filepath.suffixes:
            return KNOWN_SUFFIXES[suffix]
    return None


def load(input_filepath, output_filepath):
    handlers = list(HANDLERS.values())
    model_type = _detect_type_from_suffix(input_filepath)
    if model_type and model_type in HANDLERS:
        handlers.remove(HANDLERS[model_type])
        handlers.insert(0, HANDLERS[model_type])

    for handler in handlers:
        print("Trying" + str(handler))
        try:
            handler(input_filepath, output_filepath)
            print(f"Saved to {output_filepath}")
            return
        except Exception:
            continue

    print(f"Failed to parse the input model: {input_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Load a model from text format")
    parser.add_argument('text_filepath', type=pathlib.Path, help="File path to the model file.")
    parser.add_argument('output_filepath', type=pathlib.Path, help="Output filepath.")

    args = parser.parse_args()
    if not args.text_filepath.exists():
        parser.error(f"{args.text_filepath} is not found.")

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    load(args.text_filepath, args.output_filepath)


if __name__ == '__main__':
    main()

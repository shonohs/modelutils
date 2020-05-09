import argparse
import pathlib
from ..caffe.dump import dump_model as caffe_dump_model
from ..coreml.dump import dump_model as coreml_dump_model
from ..onnx.dump import dump_model as onnx_dump_model
from ..tensorflow.dump import dump_model as tensorflow_dump_model


HANDLERS = {'caffe': caffe_dump_model,
            'coreml': coreml_dump_model,
            'onnx': onnx_dump_model,
            'tensorflow': tensorflow_dump_model}

KNOWN_SUFFIXES = {'.mlmodel': 'coreml',
                  '.onnx': 'onnx',
                  '.pb': 'tensorflow',
                  '.prototxt': 'caffe'}


def _detect_type_from_suffix(filepath):
    return KNOWN_SUFFIXES.get(filepath.suffix)


def dump(model_filepath, output_filepath):
    handlers = list(HANDLERS.values())
    model_type = _detect_type_from_suffix(model_filepath)

    if model_type and model_type in HANDLERS:
        handlers.remove(HANDLERS[model_type])
        handlers.insert(0, HANDLERS[model_type])

    for handler in handlers:
        try:
            dumped_model = handler(model_filepath)
            break
        except Exception:
            pass

    if not dumped_model:
        raise RuntimeError(f"Unsupported file type: {model_filepath}")

    if output_filepath:
        output_filepath.write_text(dumped_model)
    else:
        print(dumped_model)


def main():
    parser = argparse.ArgumentParser(description="Dump a model into text format")
    parser.add_argument('model_filepath', type=pathlib.Path, help="File path to the model file.")
    parser.add_argument('--output', '-o', nargs='?', type=pathlib.Path, default=False, metavar='FILEPATH', help="Save to a file.")

    args = parser.parse_args()
    if not args.model_filepath.exists():
        parser.error(f"{args.model_filepath} is not found.")

    if args.output is None:
        args.output = pathlib.Path(str(args.model_filepath) + '.dump')

    if args.output and args.output.exists():
        parser.error(f"{args.output} already exists.")

    dump(args.model_filepath, args.output)


if __name__ == '__main__':
    main()

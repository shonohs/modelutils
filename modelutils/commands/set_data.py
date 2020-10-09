import argparse
import pathlib
import sys
from ..common.utils import read_input_npy, detect_type_from_suffix
from ..coreml.set_data import set_data as coreml_set_data
from ..onnx.set_data import set_data as onnx_set_data
from ..openvino.set_data import set_data as openvino_set_data
from ..tensorflow.set_data import set_data as tf_set_data
from ..tensorflowlite.set_data import set_data as tflite_set_data

HANDLERS = {'coreml': coreml_set_data,
            'onnx': onnx_set_data,
            'openvino': openvino_set_data,
            'tensorflow': tf_set_data,
            'tensorflowlite': tflite_set_data}


def set_data(model_filepath, name, model_type):
    model_type = model_type or detect_type_from_suffix(model_filepath[0])
    if not model_type:
        raise RuntimeError("Failed to detect the model type. Please specify explicitly.")

    if model_type not in HANDLERS:
        raise NotImplementedError()

    value = read_input_npy(None)
    handler = HANDLERS[model_type]
    model_filepath = model_filepath[0] if len(model_filepath) == 1 else model_filepath
    data = handler(model_filepath, name, value)

    if not data:
        raise RuntimeError("Failed to set")

    if sys.stdout.isatty():
        print("Suppressed output...")
    else:
        sys.stdout.buffer.write(data)


def main():
    parser = argparse.ArgumentParser(description="Set const data to a model")
    parser.add_argument('model_filepath', nargs='+', type=pathlib.Path)
    parser.add_argument('--name', '-n')
    parser.add_argument('--type', '-t', choices=HANDLERS.keys())

    args = parser.parse_args()

    if not args.name:
        parser.error("Please specify name")

    set_data(args.model_filepath, args.name, args.type)

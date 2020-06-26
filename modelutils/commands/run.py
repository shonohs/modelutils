import argparse
import io
import os
import pathlib
import sys
import tempfile
import numpy as np
import PIL.Image
from ..common.utils import read_input_npy, write_output_npy, detect_type_from_suffix
from ..onnx.run_model import run_model as onnx_run_model
from ..tensorflow.run_model import run_model as tf_run_model
from ..tensorflowlite.run_model import run_model as tflite_run_model

HANDLERS = {'onnx': onnx_run_model,
            'tensorflow': tf_run_model,
            'tensorflowlite': tflite_run_model}


def run(model_filepath, input_filepath, output_names, output_filepath):
    input_data = read_input_npy(input_filepath)
    assert len(input_data.shape) == 4 and input_data.shape[3] == 3
    model_type = detect_type_from_suffix(model_filepath)
    if not model_type or model_type not in HANDLERS:
        raise RuntimeError(f"Unknown extension: {model_filepath}")

    handler = HANDLERS[model_type]
    outputs = handler(model_filepath, input_data, output_names)

    if output_filepath:
        np.save(output_filepath, outputs)
    else:
        write_output_npy(outputs)


def main():
    parser = argparse.ArgumentParser("Run a model")
    parser.add_argument('model_filepath', type=pathlib.Path)
    parser.add_argument('input_filepath', type=pathlib.Path, nargs='?', help="Input npy filepath. If not given, read numpy array from stdin.")
    parser.add_argument('--output_name', nargs='*')
    parser.add_argument('--output_filepath', '-o', type=pathlib.Path)

    args = parser.parse_args()
    run(args.model_filepath, args.input_filepath, args.output_name, args.output_filepath)


if __name__ == '__main__':
    main()

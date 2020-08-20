"""Convert a tensorflow frozen graph model to TFLite."""
import argparse
import pathlib
import numpy as np
import tensorflow


def _get_graph_inputs_outputs(graph_def):
    input_names = []
    inputs_set = set()
    outputs_set = set()

    for node in graph_def.node:
        if node.op == 'Placeholder':
            input_names.append(node.name)

        for i in node.input:
            inputs_set.add(i.split(':')[0])
        outputs_set.add(node.name)

    output_names = list(outputs_set - inputs_set)
    return input_names, output_names


def convert(input_filepath, output_filepath, quantization_type, dataset_filepath):
    graph_def = tensorflow.compat.v1.GraphDef()
    graph_def.ParseFromString(input_filepath.read_bytes())

    tensorflow.compat.v1.reset_default_graph()
    tensorflow.compat.v1.disable_eager_execution()

    input_names, output_names = _get_graph_inputs_outputs(graph_def)
    nodes_by_name = {node.name: node for node in graph_def.node}
    input_tensors = [tensorflow.compat.v1.placeholder(node.attr['dtype'].type, node.attr['shape'].shape, node.name) for node in graph_def.node if node.name in input_names]
    output_tensors = [nodes_by_name[n] for n in output_names]

    converter = tensorflow.compat.v1.lite.TFLiteConverter(graph_def, input_tensors, output_tensors)
    converter.target_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_new_converter = True
    converter.allow_custom_ops = True

    if quantization_type:
        converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
        if quantization_type == 'fp16':
            converter.target_spec.supported_types = [tensorflow.float16]
        elif quantization_type == 'int8':
            def _gen_input():
                sample_inputs = np.load(dataset_filepath)
                for i in range(len(sample_inputs)):
                    yield [sample_inputs[i][np.newaxis, :]]

            converter.representative_dataset = _gen_input

    tflite_model = converter.convert()
    output_filepath.write_bytes(tflite_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pb_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)
    parser.add_argument('--quantization', '-q', choices=['fp16', 'int8'])
    parser.add_argument('--representative_dataset', '-d', type=pathlib.Path, help="filepath to a serialized npy array that contains representative dataset for INT8 quantization")

    args = parser.parse_args()
    if not args.pb_filepath.exists():
        parser.error(f"File not found: {args.pb_filepath}")

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    convert(args.pb_filepath, args.output_filepath, args.quantization, args.representative_dataset)


if __name__ == '__main__':
    main()

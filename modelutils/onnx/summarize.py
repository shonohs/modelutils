import argparse
import onnx
import onnx.numpy_helper
import numpy as np


def get_attribute(node, name):
    for attr in node.attribute:
        if attr.name == name:
            if attr.type == 1:  # FLOAT
                return attr.f
            elif attr.type == 2:  # INT
                return attr.i
            elif attr.type == 7:  # INTS
                return [i for i in attr.ints]
            else:
                raise RuntimeError(f"Unsupported type: {attr.type}")
    return None


def node_summary(node):
    summary = [node.op_type, str(node.input), '=> ' + str(node.output)]
    if node.op_type == 'Conv':
        summary += [f"(name:{node.name}, kernel_shape:{get_attribute(node, 'kernel_shape')}, group:{get_attribute(node, 'group')}, stride:{get_attribute(node, 'strides')})"]

    return summary


def print_nodes_summary(model):
    print("Nodes:")

    node_summaries = []
    for node in model.graph.node:
        node_summaries.append(node_summary(node))

    pprint_table(node_summaries)


def print_weights_summary(model):
    print("Weights:")

    weights_summaries = []
    for initializer in model.graph.initializer:
        data = onnx.numpy_helper.to_array(initializer)
        weights_summaries.append([initializer.name, str(data.shape), f"mean: {np.mean(data)}, ", f"max: {np.amax(data)},", f"min: {np.amin(data)}"])

    pprint_table(weights_summaries)


def pprint_table(table):
    max_lens = [0] * max([len(t) for t in table])
    for t in table:
        for i in range(len(t)):
            max_lens[i] = max(max_lens[i], len(t[i]))

    for t in table:
        print("".join([f"{t[i]: <{max_lens[i]}}" for i in range(len(t))]))


def summarize(filename):
    model = onnx.load(filename)

    # Get the opset version for the default opset.
    opset_version = -1
    for opset in model.opset_import:
        if opset.domain == "":
            opset_version = opset.version

    print(f"ONNX IR version: {model.ir_version}, opset version: {opset_version}")
    print_nodes_summary(model)
    print_weights_summary(model)


def main():
    parser = argparse.ArgumentParser('Get a summary of an ONNX model')
    parser.add_argument('model_filename', type=str, help='Filename for the input onnx file')

    args = parser.parse_args()
    summarize(args.model_filename)


if __name__ == '__main__':
    main()

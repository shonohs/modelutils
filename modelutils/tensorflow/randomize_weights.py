import argparse
import pathlib
import tensorflow
import numpy


def randomize_weights(input_filepath, output_filepath):
    graph_def = tensorflow.compat.v1.GraphDef()
    graph_def.ParseFromString(input_filepath.read_bytes())

    const_by_name = {node.name: node for node in graph_def.node if node.op == 'Const'}
    weights_biases_node_names = [i for node in graph_def.node for i in node.input[1:] if node.op in ['DepthwiseConv2dNative', 'Conv2D', 'BiasAdd', 'MatMul']]
    for name in weights_biases_node_names:
        print(f"Randomizing {name}...")
        node = const_by_name[name]
        value = tensorflow.make_ndarray(node.attr['value'].tensor)
        value = numpy.random.rand(*value.shape).astype(value.dtype)
        new_tensor = tensorflow.make_tensor_proto(value)
        node.attr['value'].tensor.CopyFrom(new_tensor)

    output_filepath.write_bytes(graph_def.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    randomize_weights(args.input_filepath, args.output_filepath)


if __name__ == '__main__':
    main()

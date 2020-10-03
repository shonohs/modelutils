import argparse
import pathlib


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


def cut_graph(model_filepath, output_filepath, input_names, output_names):
    """Get a minimum sub graph which has the given input names and the output names.
    Args:
        model_filepath: Input model filename.
        output_filepath: Output model filename.
        input_names: list of input tensor names. If empty, the model input names are used.
        output_names: list of output tensor names. If empty, the model output names are used.
    """
    import tensorflow
    graph_def = tensorflow.compat.v1.GraphDef()
    graph_def.ParseFromString(model_filepath.read_bytes())

    graph_input_names, graph_output_names = _get_graph_inputs_outputs(graph_def)
    input_names = input_names or graph_input_names
    output_names = output_names or graph_output_names

    required_nodes = set()
    to_visit_nodes = output_names

    # Make a dictionary {output_name: node}
    nodes_by_name = {n.name: n for n in graph_def.node}

    while to_visit_nodes:
        next_node_name = to_visit_nodes.pop()
        required_nodes.add(next_node_name)
        if next_node_name in input_names:
            continue

        next_node = nodes_by_name[next_node_name]

        for input_name in next_node.input:
            input_name = input_name.split(':')[0]
            if input_name not in to_visit_nodes and input_name not in required_nodes:
                to_visit_nodes.append(input_name)

    nodes_list = [n for n in graph_def.node if n.name in required_nodes]
    del graph_def.node[:]
    graph_def.node.extend(nodes_list)

    output_filepath.write_bytes(graph_def.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description="Cut a subgraph from a TensorFlow model")
    parser.add_argument('input_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)
    parser.add_argument('--input_names', type=str, nargs='+', default=[], help="The input names for the sub graph.")
    parser.add_argument('--output_names', type=str, nargs='+', default=[], help="The output names for the sub graph.")

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    cut_graph(args.input_filepath, args.output_filepath, args.input_names, args.output_names)


if __name__ == '__main__':
    main()

import argparse
import pathlib


def _load_graph_def(filepath):
    import tensorflow
    graph_def = tensorflow.compat.v1.GraphDef()
    graph_def.ParseFromString(filepath.read_bytes())
    return graph_def


def cat_graph(input_filepaths, output_filepath):
    model = _load_graph_def(input_filepaths[0])
    for filepath in input_filepaths[1:]:
        new_model = _load_graph_def(filepath)
        model.node.extend(new_model.node)

    output_filepath.write_bytes(model.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description="Concatenate TensorFlow graphs.")
    parser.add_argument('input_filepaths', nargs='+', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    cat_graph(args.input_filepaths, args.output_filepath)


if __name__ == '__main__':
    main()

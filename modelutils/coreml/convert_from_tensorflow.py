import argparse
import pathlib
import coremltools
import tensorflow as tf


def convert(tf_filepath, output_filepath):
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(tf_filepath.read_bytes())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')

        model = coremltools.convert(graph, source='tensorflow', inputs=[coremltools.TensorType(shape=(1, 320, 320, 3))])
        model.save(output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pb_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    convert(args.pb_filepath, args.output_filepath)


if __name__ == '__main__':
    main()

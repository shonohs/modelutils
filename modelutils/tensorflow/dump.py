import argparse
import os
import tensorflow as tf


def load_model(filename):
    try:
        graph_def = tf.compat.v1.GraphDef()
        with open(filename, 'rb') as f:
            graph_def.ParseFromString(f.read())
        return graph_def
    except Exception:
        # Try SavedModel format.
        from tensorflow.core.protobuf import saved_model_pb2
        saved_model_def = saved_model_pb2.SavedModel()
        with open(filename, 'rb') as f:
            saved_model_def.ParseFromString(f.read())
        return saved_model_def


def dump(filename, save_to_file):
    model = load_model(filename)
    if save_to_file:
        save_filename = filename + '.dump'
        if os.path.exists(save_filename):
            raise RuntimeError(f"{save_filename} already exists.")
        with open(save_filename, 'w') as f:
            f.write(str(model))
    else:
        print(model)


def main():
    parser = argparse.ArgumentParser('Dump a TensorFlow model to text')
    parser.add_argument('model_filename', type=str, help='Filename for the input TF file')
    parser.add_argument('-s', '--save', action='store_true', help='Save to .dump file')

    args = parser.parse_args()
    dump(args.model_filename, args.save)


if __name__ == '__main__':
    main()

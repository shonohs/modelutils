import argparse
import tensorflow
from google.protobuf import text_format


def load_graphdef(in_filename, out_filename):
    model = tensorflow.compat.v1.GraphDef()
    with open(in_filename, 'r') as f:
        text_format.Parse(f.read(), model)

    with open(out_filename, 'wb') as f:
        f.write(model.SerializeToString())


def load_savedmodel(in_filename, out_filename):
    from tensorflow.core.protobuf import saved_model_pb2
    saved_model_def = saved_model_pb2.SavedModel()

    with open(in_filename, 'r') as f:
        text_format.Parse(f.read(), saved_model_def)

    with open(out_filename, 'wb') as f:
        f.write(saved_model_def.SerializeToString())


def load(in_filename, out_filename):
    try:
        load_graphdef(in_filename, out_filename)
    except Exception:
        print("Failed to load graphdef. Trying SavedModel format.")
        load_savedmodel(in_filename, out_filename)


def main():
    parser = argparse.ArgumentParser('Convert a text file to ONNX model')
    parser.add_argument('text_filename', type=str, help='Filename for the input text file')
    parser.add_argument('model_filename', type=str, help='Output TF file path')

    args = parser.parse_args()
    load(args.text_filename, args.model_filename)


if __name__ == '__main__':
    main()

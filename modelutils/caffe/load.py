import argparse
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def load(in_filename, out_filename):
    net = caffe_pb2.NetParameter()
    with open(in_filename, 'r') as f:
        text_format.Parse(f.read(), net)

    with open(out_filename, 'wb') as f:
        f.write(net.SerializeToString())


def main():
    parser = argparse.ArgumentParser('Load a caffemodel from text')
    parser.add_argument('text_filename', type=str, help='Filename for the input text file')
    parser.add_argument('model_filename', type=str, help='Output caffemodel file path')

    args = parser.parse_args()
    load(args.text_filename, args.model_filename)


if __name__ == '__main__':
    main()

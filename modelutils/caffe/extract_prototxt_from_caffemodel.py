"""Extract prototxt file from caffemodel file.
"""
import argparse
import pathlib
from caffe.proto import caffe_pb2


def extract_prototxt_from_caffemodel(caffemodel_filepath, output_prototxt_filepath):
    model = caffe_pb2.NetParameter()
    model.ParseFromString(caffemodel_filepath.read_bytes())

    # Remove weights from the model.
    for layer in model.layer:
        layer.ClearField('phase')
        if hasattr(layer, 'blobs'):
            del layer.blobs[:]

    # Remove Split layers from the model.
    top_aliases = {}
    split_layers = [layer for layer in model.layer if layer.type == 'Split']
    non_split_layers = [layer for layer in model.layer if layer.type != 'Split']
    for layer in split_layers:
        assert len(layer.bottom) == 1
        for top in layer.top:
            top_aliases[top] = layer.bottom[0]
    del model.layer[:]
    model.layer.extend(non_split_layers)

    # Rename the aliases.
    for layer in model.layer:
        for i, bottom in enumerate(layer.bottom):
            if bottom in top_aliases:
                layer.bottom[i] = top_aliases[bottom]

    output_prototxt_filepath.write_text(str(model))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffemodel_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    extract_prototxt_from_caffemodel(args.caffemodel_filepath, args.output_filepath)


if __name__ == '__main__':
    main()

import argparse
import pathlib
import sys
import flatbuffers


def strip_tensor_name(input_filepath, output_filepath):
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from tflite.Model import ModelT, Model

    input_binary = input_filepath.read_bytes()
    model = ModelT.InitFromObj(Model.GetRootAsModel(input_binary, 0))

    assert len(model.subgraphs) == 1
    graph = model.subgraphs[0]

    for i, tensor in enumerate(graph.tensors):
        if i not in graph.inputs + graph.outputs:
            tensor.name = str(i).encode('utf-8')

    builder = flatbuffers.Builder(len(input_binary))
    offset = model.Pack(builder)
    builder.Finish(offset, file_identifier=b'\x54\x46\x4C\x33')
    output_filepath.write_bytes(builder.Output())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if not args.input_filepath.exists():
        parser.error(f"Input file not found: {args.input_filepath}")

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    strip_tensor_name(args.input_filepath, args.output_filepath)


if __name__ == '__main__':
    main()

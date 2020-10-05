from google.protobuf import text_format


def load_model(in_filepath, out_filepath):
    import coremltools
    model = coremltools.proto.Model_pb2.Model()
    text_format.Parse(in_filepath.read_text(), model)
    out_filepath.write_bytes(model.SerializeToString())

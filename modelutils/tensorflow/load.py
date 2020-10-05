from google.protobuf import text_format


def load_graphdef(in_filepath, out_filepath):
    import tensorflow
    model = tensorflow.compat.v1.GraphDef()
    text_format.Parse(in_filepath.read_text())
    out_filepath.write_bytes(model.SerializeToString())


def load_savedmodel(in_filepath, out_filepath):
    from tensorflow.core.protobuf import saved_model_pb2
    model = saved_model_pb2.SavedModel()
    text_format.Parse(in_filepath.read_text())
    out_filepath.write_bytes(model.SerializeToString())


def load_model(in_filename, out_filename):
    try:
        load_graphdef(in_filename, out_filename)
    except Exception:
        load_savedmodel(in_filename, out_filename)

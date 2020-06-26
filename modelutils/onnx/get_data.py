import onnx
import onnx.numpy_helper


def get_data(model_filepath, name):
    model = onnx.load(model_filepath)
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return onnx.numpy_helper.to_array(initializer)

    return None

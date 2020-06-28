import onnx
import onnx.numpy_helper


def get_data(model_filepath, name):
    model = onnx.load(model_filepath)
    if name:
        for initializer in model.graph.initializer:
            if initializer.name == name:
                return onnx.numpy_helper.to_array(initializer)

    return {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}

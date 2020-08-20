import pathlib
import sys
import numpy as np


def get_data(model_filepath, name):
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))
    from tflite.Model import ModelT, Model

    model = ModelT.InitFromObj(Model.GetRootAsModel(model_filepath.read_bytes(), 0))

    assert len(model.subgraphs) == 1
    graph = model.subgraphs[0]

    tensors_by_name = {tensor.name.decode('utf-8'): tensor for tensor in graph.tensors if tensor.buffer > 1 and model.buffers[tensor.buffer].data is not None}

    if name:
        return _get_value_for_tensor(graph, tensors_by_name[name])
    return {name: _get_value_for_tensor(model, value) for name, value in tensors_by_name.items()}


def _get_value_for_tensor(model, tensor):
    from tflite.TensorType import TensorType
    data = model.buffers[tensor.buffer].data
    dtype = {TensorType.FLOAT32: np.float32,
             TensorType.FLOAT16: np.float16,
             TensorType.INT32: np.int32,
             TensorType.INT8: np.int8}[tensor.type]
    value = np.frombuffer(bytes(data), dtype=dtype).reshape(tensor.shape)

    if tensor.quantization:
        scale = tensor.quantization.scale
        zero_point = tensor.quantization.zeroPoint
        if scale is not None and zero_point is not None:
            dims = [i for i in range(value.ndim) if i != tensor.quantization.quantizedDimension]
            scale = np.expand_dims(np.array(scale), axis=dims)
            zero_point = np.expand_dims(np.array(zero_point), axis=dims)
            value = (value - zero_point) * scale
    return value

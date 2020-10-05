import numpy as np


def _get_nn_layers(model):
    if model.WhichOneof('Type') in ['pipeline', 'pipelineClassifier', 'pipelineRegressor']:
        layers = []
        for m in model.pipeline.models:
            layers.extend(_get_nn_layers(m))
    else:
        layers = model.neuralNetwork.layers or model.neuralNetworkClassifier.layers or model.neuralNetworkRegressor.layers
    return layers


def _read_value(data, shape):
    if data.floatValue:
        return np.array(data.floatValue, dtype=np.float32).reshape(shape)
    elif data.float16Value:
        return np.frombuffer(np.float16Value, dtype=np.float16).reshape(shape)
    else:
        raise RuntimeError(f"Unsupported data type: {data}")


def get_data(model_filepath, name):
    import coremltools
    model = coremltools.models.MLModel(str(model_filepath)).get_spec()

    layers = _get_nn_layers(model)

    data_dict = {}
    for layer in layers:
        layer_type = layer.WhichOneof('layer')
        if layer_type == 'convolution':
            conv = layer.convolution
            data_dict[layer.name + '/' + 'weights'] = _read_value(conv.weights, [conv.kernelChannels, conv.outputChannels, *conv.kernelSize])
            if layer.convolution.hasBias:
                data_dict[layer.name + '/' + 'bias'] = _read_value(conv.bias, [conv.outputChannels])
        elif layer_type == 'loadConstant':
            data_dict[layer.name] = _read_value(layer.loadConstant.data, layer.loadConstant.shape)

    if name:
        return {name: data_dict[name]}

    return data_dict

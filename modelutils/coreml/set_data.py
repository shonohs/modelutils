import numpy as np


def _get_nn_layers(model):
    if model.WhichOneof('Type') in ['pipeline', 'pipelineClassifier', 'pipelineRegressor']:
        layers = []
        for m in model.pipeline.models:
            layers.extend(_get_nn_layers(m))
    else:
        layers = model.neuralNetwork.layers or model.neuralNetworkClassifier.layers or model.neuralNetworkRegressor.layers
    return layers


def _write_value(data, value):
    if value.dtype == np.float32:
        del data.floatValue[:]
        data.floatValue.extend(value.flatten())
    elif value.dtype == np.float16:
        data.float16Value = value.flatten().tobytes()
    else:
        raise NotImplementedError


def set_data(model_filepath, name, value):
    import coremltools
    model = coremltools.models.MLModel(str(model_filepath)).get_spec()

    layers = _get_nn_layers(model)

    for layer in layers:
        layer_type = layer.WhichOneof('layer')
        if layer_type == 'convolution':
            conv = layer.convolution
            if name == layer.name + '/' + 'weights':
                conv.kernelChannels = value.shape[1]
                conv.outputChannels = value.shape[0]
                assert len(conv.kernelSize) == 2
                del conv.kernelSize[:]
                conv.kernelSize.extend(value.shape[2:])
                _write_value(conv.weights, value)
                return model.SerializeToString()
            elif name == layer.name + '/' + 'bias':
                conv.outputChannels = value.shape[0]
                _write_value(conv.bias, value)
                return model.SerializeToString()
        elif layer_type == 'loadConstant' and layer.name == name:
            del layer.loadConstant.shape[:]
            layer.loadConstant.shape.extend(value.shape)
            _write_value(layer.loadConstant.data, value)
            return model.SerializeToString()

    return None

import xml.etree.ElementTree as ET
import numpy as np


_ELEMENT_TYPE_TO_DTYPE = {'f32': np.float32,
                          'i64': np.int64,
                          'i32': np.int32,
                          'i8': np.int8}

def get_const_weight(weights, layer):
    data_node = layer.find('data')
    shape = data_node.get('shape').strip()
    if shape:
        shape = [int(s) for s in shape.split(',')]
    else:
        shape = [1]

    offset = int(data_node.get('offset'))
    size = int(data_node.get('size'))
    dtype = _ELEMENT_TYPE_TO_DTYPE[data_node.get('element_type')]
    return np.frombuffer(weights[offset:offset+size], dtype=dtype).reshape(*shape)


def get_data(model_filepath, name):
    assert len(model_filepath) == 2, "OpenVino requires two model files: .xml and .bin"

    tree = ET.parse(model_filepath[0])
    weights = model_filepath[1].read_bytes()
    const_layers = {layer.get('name'): layer for layer in tree.getroot().find('layers').findall('layer') if layer.get('type') == 'Const'}

    if name:
        return {name: get_const_weight(weights, const_layers[name])}
    else:
        return {n: get_const_weight(weights, layer) for n, layer in const_layers.items()}

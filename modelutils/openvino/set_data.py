import xml.etree.ElementTree as ET
import numpy as np

_ELEMENT_TYPE_TO_DTYPE = {'f32': np.float32,
                          'i64': np.int64,
                          'i32': np.int32,
                          'i8': np.int8}


def set_data(model_filepath, name, value):
    assert len(model_filepath) == 2, "OpenVino requires two model files: .xml and .bin"

    tree = ET.parse(model_filepath[0])
    weights = bytearray(model_filepath[1].read_bytes())

    for layer in tree.getroot().find('layers').findall('layer'):
        if layer.get('type') == 'Const' and layer.get('name') == name:
            data_node = layer.find('data')
            shape = data_node.get('shape').strip()
            shape = [int(s) for s in shape.split(',')] if shape else [1]
            offset = int(data_node.get('offset'))
            size = int(data_node.get('size'))
            dtype = _ELEMENT_TYPE_TO_DTYPE[data_node.get('element_type')]
            binary = value.astype(dtype).tobytes()
            if len(binary) != size or shape != list(value.shape):
                raise RuntimeError(f"Changing size is not supported yet. {shape} vs {value.shape}, {len(binary)} vs {size}")

            weights[offset:offset+size] = binary
            return weights

    raise RuntimeError

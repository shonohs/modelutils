import argparse
import xml.etree.ElementTree as ET
import numpy as np


def pprint_table(table):
    max_lens = [0] * max([len(t) for t in table])
    for t in table:
        for i in range(len(t)):
            max_lens[i] = max(max_lens[i], len(t[i]))

    for t in table:
        print("".join([f"{t[i]: <{max_lens[i]}}" for i in range(len(t))]))


def get_weight_info(node_id, data_node, weights):
    shape = data_node.get('shape').strip()
    if shape:
        shape = [int(s) for s in shape.split(',')]
    else:
        shape = [1]

    element_type = data_node.get('element_type')
    offset = int(data_node.get('offset'))
    dtype = {'f32': np.float32, 'i32': np.int32, 'i64': np.int64}[element_type]
    if weights:
        array = np.frombuffer(weights, dtype, count=np.prod(shape), offset=offset)
        min_value = np.amin(array)
        max_value = np.amax(array)
        mean_value = np.mean(array)
        return [f"{node_id}, ", f"{element_type}, ", f"{shape}, ", f"mean: {mean_value}", f"max: {max_value}", f"min: {min_value}"]
    else:
        return [f"{node_id}, ", f"{element_type}, ", f"{shape}, "]


def summarize(filename, bin_filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    if bin_filename:
        with open(bin_filename, 'rb') as f:
            weights = f.read()
    else:
        weights = None

    lines = []
    for layer in root.find('layers').findall('layer'):
        if layer.get('type') == 'Const':
            data_node = layer.find('data')
            lines.append(get_weight_info(layer.get('id'), data_node, weights))

    pprint_table(lines)


def main():
    parser = argparse.ArgumentParser('Get a summary of an OpenVino model')
    parser.add_argument('xml_filename', type=str, help='Filename for the input openvino XML file')
    parser.add_argument('bin_filename', nargs='?', help="Filename for the weights file")

    args = parser.parse_args()
    summarize(args.xml_filename, args.bin_filename)


if __name__ == '__main__':
    main()

import argparse
import pathlib
import numpy as np
import onnx
import onnx.numpy_helper
import tensorflow as tf


class Converter:
    def __init__(self, model):
        self.onnx_model = model
        self._tensors = {}
        self._initializers = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}

    def convert(self):
        self.handle_inputs()

        self.add_tensor('paddings1', tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
        self.add_tensor('paddings2', tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))

        for node in self.onnx_model.graph.node:
            method = getattr(self, f'handle_{node.op_type.lower()}', None)
            if not method:
                raise RuntimeError(f"Undefined handler: {node.op_type.lower()}. {node}")
            method(node)

    def handle_inputs(self):
        for i in self.onnx_model.graph.input:
            shape = [d.dim_value or 1 for d in i.type.tensor_type.shape.dim]
            assert len(shape) == 4
            shape = [shape[0], shape[2], shape[3], shape[1]]  # => NHWC
            placeholder = tf.compat.v1.placeholder(tf.float32, shape=shape, name=i.name)
            self.add_tensor(i.name, placeholder)
            print(f"Added input: {i.name} shape={shape}")

    def handle_add(self, node):
        inputs = [self.get_tensor(i) for i in node.input]
        if len(inputs) == 2:
            tensor = tf.math.add(inputs[0], inputs[1], name=node.name)
        else:
            tensor = tf.math.add_n(inputs, name=node.name)
        self.add_tensor(node.output[0], tensor)

    def handle_clip(self, node):
        attrs = {a.name: a for a in node.attribute}
        if attrs['min'].f == 0 and attrs['max'].f == 6:
            tensor = tf.nn.relu6(self.get_tensor(node.input[0]), name=node.name)
        else:
            raise RuntimeError
        self.add_tensor(node.output[0], tensor)

    def handle_conv(self, node):
        attrs = {a.name: a for a in node.attribute}
        strides = attrs['strides'].ints[0] if 'strides' in attrs else 1
        groups = attrs['group'].i if 'group' in attrs else 1
        pads = attrs['pads'].ints[0] if 'pads' in attrs else 0

        tensor = self.get_tensor(node.input[0])
        if pads > 0:
            paddings = self.get_tensor(f'paddings{pads}')
            tensor = tf.compat.v1.pad(tensor, paddings, 'CONSTANT')

        if groups > 1:
            strides = [1, strides, strides, 1] if strides > 1 else [1, 1, 1, 1]
            tensor = tf.compat.v1.nn.depthwise_conv2d(tensor,
                                                      self.get_tensor(node.input[1], transpose=(2, 3, 0, 1), const_name=node.name + '/weights'),
                                                      strides=strides, padding='VALID', name=node.name + '/conv2d')
        else:
            tensor = tf.compat.v1.nn.conv2d(tensor,
                                            self.get_tensor(node.input[1], transpose=(2, 3, 1, 0), const_name=node.name + '/biases'),
                                            strides=strides, padding='VALID', name=node.name + '/conv2d')
        if len(node.input) > 2:
            tensor = tf.nn.bias_add(tensor, self.get_tensor(node.input[2]), name=node.name + '/biasadd')

        self.add_tensor(node.output[0], tensor)

    def handle_div(self, node):
        tensor = tf.math.divide(self.get_tensor(node.input[0]), self.get_tensor(node.input[1]), name=node.name)
        self.add_tensor(node.output[0], tensor)

    def handle_flatten(self, node):
        tensor = tf.compat.v1.squeeze(self.get_tensor(node.input[0]), axis=(1, 2), name=node.name)
        self.add_tensor(node.output[0], tensor)

    def handle_gemm(self, node):
        tensor = tf.linalg.matmul(self.get_tensor(node.input[0]), self.get_tensor(node.input[1], transpose=(1, 0)), name=node.name + '/matmul')
        tensor = tf.nn.bias_add(tensor, self.get_tensor(node.input[2]), name=node.name + '/biasadd')
        self.add_tensor(node.output[0], tensor)

    def handle_globalaveragepool(self, node):
        tensor = tf.compat.v1.reduce_mean(self.get_tensor(node.input[0]), axis=(1,2), keepdims=True, name=node.name)
        #tensor = tf.keras.layers.GlobalAveragePooling2D()(self.get_tensor(node.input[0]))
        self.add_tensor(node.output[0], tensor)

    def handle_mul(self, node):
        tensor = tf.math.multiply(self.get_tensor(node.input[0]), self.get_tensor(node.input[1]), name=node.name)
        self.add_tensor(node.output[0], tensor)

    def handle_softmax(self, node):
        tensor = tf.compat.v1.math.softmax(self.get_tensor(node.input[0]), name=node.name)
        self.add_tensor(node.output[0], tensor)

    def get_tensor(self, name, transpose=None, const_name=None):
        if name in self._tensors:
            assert not transpose
            return self._tensors[name]

        if name in self._initializers:
            value = self._initializers[name]
            if transpose:
                value = value.transpose(transpose)
            tensor = tf.compat.v1.constant(value, name=name)
            self.add_tensor(name, tensor)
            return tensor

        raise RuntimeError(f"Unknown tensor: {name}")

    def add_tensor(self, name, tensor):
        assert name not in self._tensors
        self._tensors[name] = tensor

def strip_unused_const(filepath):
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(filepath.read_bytes())

    input_set = set()
    for node in graph_def.node:
        for i in node.input:
            input_set.add(i)

    new_nodes = [node for node in graph_def.node if node.op != 'Const' or node.name in input_set]
    del graph_def.node[:]
    graph_def.node.extend(new_nodes)

    filepath.write_bytes(graph_def.SerializeToString())

def convert(onnx_filepath, output_filepath):

    onnx_model = onnx.load(onnx_filepath)

    graph = tf.Graph()
    converter = Converter(onnx_model)
    with graph.as_default():
        converter.convert()

    if output_filepath:
        tf.io.write_graph(graph, str(output_filepath.parent), str(output_filepath.name), as_text=False)
        strip_unused_const(output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', nargs='?', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath and args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")
    convert(args.onnx_filepath, args.output_filepath)


if __name__ == '__main__':
    main()

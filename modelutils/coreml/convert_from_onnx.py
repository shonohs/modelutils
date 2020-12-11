import argparse
import pathlib
import coremltools
import onnx
import onnx.numpy_helper


class Converter:
    def __init__(self, model, num_classes):
        self.onnx_model = model
        self.num_classes = num_classes
        self._tensors = {}
        self._initializers = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}

    def convert(self):
        self.model = coremltools.proto.Model_pb2.Model()
        self.model.specificationVersion = 1
        self.model.description.predictedFeatureName = 'classLabel'
        self.model.description.predictedProbabilitiesName = 'model_output'

        dict_output = self.model.description.output.add()
        dict_output.name = 'model_output'
        dict_output.type.dictionaryType.stringKeyType.MergeFromString(b'')

        label_output = self.model.description.output.add()
        label_output.name = 'classLabel'
        label_output.type.stringType.MergeFromString(b'')

        self.handle_inputs()
        self.network = self.model.neuralNetworkClassifier

        self.network.stringClassLabels.vector.extend([str(i) for i in range(self.num_classes)])

        for node in self.onnx_model.graph.node:
            method = getattr(self, f'handle_{node.op_type.lower()}', None)
            if not method:
                raise RuntimeError(f"Undefined handler: {node.op_type.lower()}. {node}")
            method(node)

        return self.model

    def handle_inputs(self):
        for i in self.onnx_model.graph.input:
            shape = [d.dim_value or 1 for d in i.type.tensor_type.shape.dim]
            assert len(shape) == 4
            new_input = self.model.description.input.add()
            new_input.name = i.name
            new_input.type.imageType.width = shape[3]
            new_input.type.imageType.height = shape[2]
            new_input.type.imageType.colorSpace = coremltools.proto.FeatureTypes_pb2.ImageFeatureType.ColorSpace.RGB

            print(f"Added input: {i.name} shape={shape}")

    def handle_outputs(self):
        for o in self.onnx_model.graph.output:
            new_output = self.model.description.output.add()
            new_output.name = o.name
            new_output.type.multiArrayType.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32

    def handle_add(self, node):
        value = self.get_const(node.input[1])
        if value is not None:
            layer = self.add_layer(node.name, node.input[0:1], node.output)
            assert len(value.shape) == 0 or value.shape == (1,)
            layer.add.alpha = float(value)
        else:
            layer = self.add_layer(node.name, node.input, node.output)
            layer.add.MergeFromString(b'')

    def handle_clip(self, node):
        attrs = {a.name: a for a in node.attribute}
        if attrs['min'].f == 0 and attrs['max'].f == 6:
            threshold_layer = self.add_layer(node.name + '/0', node.input, [node.output[0] + '/0'])
            threshold_layer.unary.type = coremltools.proto.NeuralNetwork_pb2.UnaryFunctionLayerParams.THRESHOLD
            threshold_layer.unary.epsilon = 1e-8
            threshold_layer.unary.scale = 1.0

            threshold_layer2 = self.add_layer(node.name + '/1', [node.output[0] + '/0'], [node.output[0] + '/1'])
            threshold_layer2.unary.type = coremltools.proto.NeuralNetwork_pb2.UnaryFunctionLayerParams.THRESHOLD
            threshold_layer2.unary.alpha = -6.0
            threshold_layer2.unary.epsilon = 1e-8
            threshold_layer2.unary.scale = -1.0

            linear_activation_layer = self.add_layer(node.name, [node.output[0] + '/1'], node.output)
            linear_activation_layer.activation.linear.alpha = -1.0
        else:
            raise RuntimeError

    def handle_conv(self, node):
        attrs = {a.name: a for a in node.attribute}
        strides = attrs['strides'].ints if 'strides' in attrs else [1, 1]
        groups = attrs['group'].i if 'group' in attrs else 1
        pads = attrs['pads'].ints[0] if 'pads' in attrs else 0
        kernel_shape = attrs['kernel_shape'].ints

        weights = self.get_const(node.input[1])
        biases = self.get_const(node.input[2]) if len(node.input) > 2 else None
        layer = self.add_layer(node.name, node.input[0:1], node.output)
        layer.convolution.outputChannels = weights.shape[0]
        layer.convolution.kernelChannels = weights.shape[1]
        layer.convolution.kernelSize.extend(kernel_shape)
        layer.convolution.stride.extend(strides)
        layer.convolution.hasBias = biases is not None

        layer.convolution.valid.MergeFromString(b'')
        if pads > 0:
            for i in range(2):
                border = layer.convolution.valid.paddingAmounts.borderAmounts.add()
                border.startEdgeSize = pads
                border.endEdgeSize = pads

        layer.convolution.weights.floatValue.extend(weights.flatten())
        if biases is not None:
            layer.convolution.bias.floatValue.extend(biases.flatten())

        if groups > 1:
            layer.convolution.nGroups = groups

    def handle_div(self, node):
        layer = self.add_layer(node.name, node.input[0:1], node.output)
        value = self.get_const(node.input[1])
        assert len(value.shape) == 0 or value.shape == (1,)
        layer.multiply.alpha = 1 / float(value)

    def handle_flatten(self, node):
        layer = self.add_layer(node.name, node.input, node.output)
        layer.flatten.MergeFromString(b'')

    def handle_gemm(self, node):
        weights = self.get_const(node.input[1])
        biases = self.get_const(node.input[2])

        layer = self.add_layer(node.name, node.input[0:1], node.output)
        layer.innerProduct.hasBias = True
        layer.innerProduct.inputChannels = weights.shape[1]
        layer.innerProduct.outputChannels = weights.shape[0]
        layer.innerProduct.weights.floatValue.extend(weights.flatten())
        layer.innerProduct.bias.floatValue.extend(biases.flatten())

    def handle_globalaveragepool(self, node):
        layer = self.add_layer(node.name, node.input, node.output)
        layer.pooling.type = coremltools.proto.NeuralNetwork_pb2.PoolingLayerParams.AVERAGE
        layer.pooling.globalPooling = True
        layer.pooling.valid.MergeFromString(b'')

    def handle_mul(self, node):
        value = self.get_const(node.input[1])
        if value is not None:
            layer = self.add_layer(node.name, node.input[0:1], node.output)
            assert len(value.shape) == 0 or value.shape == (1,)
            layer.multiply.alpha = float(value)
        else:
            layer = self.add_layer(node.name, node.input, node.output)
            layer.multiply.MergeFromString(b'')

    def handle_softmax(self, node):
        layer = self.add_layer(node.name, node.input, node.output)
        layer.softmax.MergeFromString(b'')

    def add_layer(self, name, inputs, outputs):
        layer = self.network.layers.add()
        layer.name = name
        layer.input.extend(inputs)
        layer.output.extend(outputs)
        return layer

    def get_const(self, name):
        return self._initializers.get(name)


def convert(onnx_filepath, output_filepath, num_classes):
    onnx_model = onnx.load(onnx_filepath)

    converter = Converter(onnx_model, num_classes)
    model = converter.convert()

    if output_filepath:
        output_filepath.write_bytes(model.SerializeToString())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', nargs='?', type=pathlib.Path)
    parser.add_argument('--num_classes', type=int, default=1000)

    args = parser.parse_args()

    if args.output_filepath and args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    convert(args.onnx_filepath, args.output_filepath, args.num_classes)


if __name__ == '__main__':
    main()

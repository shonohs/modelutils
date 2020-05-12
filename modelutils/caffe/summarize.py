from google.protobuf import text_format
from ..common.model import Model


class CaffeModelParser:
    def __init__(self, prototxt_filepath, caffemodel_filepath):
        from caffe.proto import caffe_pb2
        self.net = caffe_pb2.NetParameter()
        text_format.Parse(prototxt_filepath.read_text(), self.net)
        self.weights = caffe_pb2.NetParameter()
        self.weights.ParseFromString(caffemodel_filepath.read_bytes())

    def parse(self):
        model = Model()
        top_name_mapper = {}
        weights_index = 0
        for layer in self.net.layer:
            inputs = [top_name_mapper[i] for i in layer.bottom]

            names = layer.top
            for i, name in enumerate(names):
                new_name = name + '_' if name in top_name_mapper else name
                top_name_mapper[name] = new_name
                names[i] = new_name

            if layer.type in ['Convolution', 'InnerProduct']:
                weights = self._find_weights(layer.name)
                for i, w in enumerate(weights):
                    n = f"weights_{weights_index+i}"
                    model.add_data(n, w)
                    inputs.append(n)
                weights_index += len(weights)

            if layer.type != 'Input':
                for name in names:
                    model.add_node(layer.type, name, inputs, {'name': layer.name})
        return model

    def _find_weights(self, name):
        import caffe
        for layer in self.weights.layer:
            if layer.name == name:
                return [caffe.io.blobproto_to_array(blob) for blob in layer.blobs]


def parse_model(prototxt_filepath, caffemodel_filepath):
    parser = CaffeModelParser(prototxt_filepath, caffemodel_filepath)
    return parser.parse()

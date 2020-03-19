import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

INPUT_TENSOR_NAME = 'Placeholder:0'


def get_graph_inputs_outputs(graph_def):
    input_names = []
    inputs_set = set()
    outputs_set = set()

    for node in graph_def.node:
        if node.op == 'Placeholder':
            input_names.append(node.name)

        for i in node.input:
            inputs_set.add(i.split(':')[0])
        outputs_set.add(node.name)

    output_names = list(outputs_set - inputs_set)
    return input_names, output_names


def preprocess_inputs(image_filename, input_shape, is_bgr=True, normalize_inputs=False, subtract_inputs=[]):
    image = Image.open(image_filename)
    image = image.resize(input_shape, Image.ANTIALIAS)
    image = image.convert('RGB') if image.mode != 'RGB' else image
    image = np.asarray(image, dtype=np.float32)

    if subtract_inputs:
        assert len(subtract_inputs) == 3
        image -= np.array(subtract_inputs, dtype=np.float32)

    image = image[:, :, (2, 1, 0)] if is_bgr else image  # RGB -> BGR
    image = image[np.newaxis, :]

    if normalize_inputs:
        image /= 255

    return image


def dump_outputs(name, shape, data):
    print('-- {} {}'.format(name, shape))
    for i in range(len(data)):
        print('{}: {}'.format(i, data[i]))


def run(model_filename, image_filename, output_names, normalize_inputs, subtract_inputs, is_bgr):
    graph_def = tf.compat.v1.GraphDef()
    with open(model_filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    graph_in, graph_out = get_graph_inputs_outputs(graph_def)

    assert len(graph_in) == 1
    input_tensor_name = graph_in[0] + ':0'
    if not output_names:
        output_names = [o + ':0' for o in graph_out]

    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name(input_tensor_name).shape.as_list()

    inputs = preprocess_inputs(image_filename, input_tensor_shape[1:3], is_bgr, normalize_inputs, subtract_inputs)

    with tf.compat.v1.Session() as sess:
        tensors = [sess.graph.get_tensor_by_name(o) for o in output_names]
        outputs = sess.run(tensors, {input_tensor_name: inputs})

    for i, name in enumerate(output_names):
        if len(outputs[i].shape) == 4:
            outputs[i] = outputs[i].transpose((0, 3, 1, 2))
        dump_outputs(name, outputs[i].shape, outputs[i].flatten())


def main():
    parser = argparse.ArgumentParser('Run a TensorFlow model with ONNX Runtime')
    parser.add_argument('pb_filename', type=str, help='Filename for the pb file')
    parser.add_argument('image_filename', type=str, help='Filename for the input image')
    parser.add_argument('--output_name', type=str, nargs='+', default=[], help='Blob name to be extracted')
    parser.add_argument('--normalize_inputs', action='store_true', help="Normalize the input to [0-1] range")
    parser.add_argument('--subtract_inputs', nargs='+', help="Subtract specified values from RGB inputs. ex) --subtract_inputs 123 117 104")
    parser.add_argument('--bgr', action='store_true', help="Use BGR instead of RGB")

    args = parser.parse_args()
    run(args.pb_filename, args.image_filename, args.output_name, args.normalize_inputs, args.subtract_inputs, args.bgr)


if __name__ == '__main__':
    main()

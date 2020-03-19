import argparse
import os
import tempfile
import onnx
import onnxruntime
from PIL import Image
import numpy as np


def add_output_node(input_model_filename, output_model_filename, output_node_names):
    model = onnx.ModelProto()
    with open(input_model_filename, 'rb') as f:
        model.ParseFromString(f.read())

    existing_output_names = set([o.name for o in model.graph.output])
    for output_name in set(output_node_names) - existing_output_names:
        model.graph.output.append(onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None))

    with open(output_model_filename, 'wb') as f:
        f.write(model.SerializeToString())


def crop_image_center(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = left + new_size
    bottom = top + new_size
    return image.crop((left, top, right, bottom))


def preprocess_inputs(image_filename, input_shape, input_type, is_bgr=True, normalize_inputs=False, subtract_inputs=[], center_crop=False):
    image = Image.open(image_filename)
    if center_crop:
        image = crop_image_center(image)
    print(image.size)
    image = image.resize(input_shape, Image.ANTIALIAS)
    image = image.convert('RGB') if image.mode != 'RGB' else image
    image = np.asarray(image, dtype=np.float32)

    if subtract_inputs:
        assert len(subtract_inputs) == 3
        image -= np.array(subtract_inputs, dtype=np.float32)

    image = image[:, :, (2, 1, 0)] if is_bgr else image  # RGB -> BGR
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = image.astype(np.float16) if input_type == 'tensor(float16)' else image

    if normalize_inputs:
        image /= 255
    return image


def dump_outputs(name, shape, data):
    print('-- {} {}'.format(name, shape))
    for i in range(len(data)):
        print('{}: {}'.format(i, data[i]))


def run(model_filename, image_filename, output_names, normalize_inputs, subtract_inputs, is_bgr, enable_profiling, image_size, center_crop):
    session_options = onnxruntime.SessionOptions()
    session_options.enable_profiling = enable_profiling

    if output_names:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_model_filename = os.path.join(tempdir, 'model.onnx')
            add_output_node(model_filename, temp_model_filename, output_names)
            sess = onnxruntime.InferenceSession(temp_model_filename, sess_options=session_options)
    else:
        sess = onnxruntime.InferenceSession(model_filename, sess_options=session_options)
        output_names = [o.name for o in sess.get_outputs()]

    # TODO: Read input format from ONNX model
    image_shape = sess.get_inputs()[0].shape[2:]
    if image_size:
        image_shape = [int(image_size[0]), int(image_size[0])]
    image = preprocess_inputs(image_filename, image_shape, sess.get_inputs()[0].type, is_bgr, normalize_inputs, subtract_inputs, center_crop)

    outputs = sess.run(output_names, {sess.get_inputs()[0].name: image})

    for i, output_name in enumerate(output_names):
        if isinstance(outputs[i], list):
            assert len(outputs[i]) == 1
            outputs[i] = outputs[i][0]

        if isinstance(outputs[i], dict):
            keys = sorted(list(outputs[i].keys()))
            outputs[i] = np.array([outputs[i][k] for k in keys])

        dump_outputs(output_name, outputs[i].shape, outputs[i].flatten())


def main():
    parser = argparse.ArgumentParser('Run a ONNX model with ONNX Runtime')
    parser.add_argument('onnx_filename', type=str, help='Filename for the ONNX file')
    parser.add_argument('image_filename', type=str, help='Filename for the input image')
    parser.add_argument('--output_name', type=str, nargs='+', default=[], help='Blob name to be extracted')
    parser.add_argument('--normalize_inputs', action='store_true', help="Normalize the input to [0-1] range")
    parser.add_argument('--subtract_inputs', nargs='+', help="Subtract specified values from RGB inputs. ex) --subtract_inputs 123 117 104")
    parser.add_argument('--bgr', action='store_true', help="Use BGR instead of RGB")
    parser.add_argument('--center_crop', action='store_true', help="Preprocess the input image by center cropping")
    parser.add_argument('--enable_profiling', action='store_true', default=False, help="Enable ONNX profiling")
    parser.add_argument('--image_size', nargs=1, help="Size of input image")

    args = parser.parse_args()
    run(args.onnx_filename, args.image_filename, args.output_name, args.normalize_inputs, args.subtract_inputs, args.bgr, args.enable_profiling, args.image_size, args.center_crop)


if __name__ == '__main__':
    main()

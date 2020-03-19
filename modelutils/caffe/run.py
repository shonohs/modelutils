import argparse
import caffe
import numpy as np
from PIL import Image


def caffe_forward(net, image_filename, blob_names):
    assert(len(net.inputs) == 1)

    if not blob_names:
        blob_names = net.outputs

    input_blob = net.blobs[net.inputs[0]]

    # Open and preprocess the image.
    image = Image.open(image_filename)
    image = image.resize(input_blob.shape[2:], Image.ANTIALIAS)
    image = np.array(image, dtype=np.float32)
    image = image.transpose((2, 0, 1))  # (H x W x C) => (C x H x W)
    image = image[(2, 1, 0), :, :]  # RGB -> BGR
    image = image[np.newaxis, :]

    input_blob.reshape(*image.shape)
    input_blob.data[...] = image

    net.forward()

    return {blob_name: net.blobs[blob_name].data for blob_name in blob_names}


def dump_outputs(name, shape, data):
    print('-- {} {}'.format(name, shape))
    for i in range(len(data)):
        print('{}: {}'.format(i, data[i]))


def run(deploy_filename, weights_filename, image_filename, output_names):
    net = caffe.Net(deploy_filename, caffe.TEST, weights=weights_filename)

    outputs = caffe_forward(net, image_filename, output_names)

    for output_name in outputs:
        dump_outputs(output_name, outputs[output_name].shape, outputs[output_name].flatten())


def main():
    parser = argparse.ArgumentParser('Forward run with a caffe model')
    parser.add_argument('prototxt_filename', type=str, help='Filename for the prototxt file')
    parser.add_argument('caffemodel_filename', type=str, help='Filename for the caffemodel file')
    parser.add_argument('image_filename', type=str, help='Filename for the input image')
    parser.add_argument('--output_name', type=str, nargs='+', default=['loss'], help='Blob name to be extracted')

    args = parser.parse_args()
    run(args.prototxt_filename, args.caffemodel_filename, args.image_filename, args.output_name)


if __name__ == '__main__':
    main()

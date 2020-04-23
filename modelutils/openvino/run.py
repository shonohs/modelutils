import argparse
import PIL.Image
import numpy as np
from openvino.inference_engine import IENetwork, IECore


def dump_outputs(name, shape, data):
    print('-- {} {}'.format(name, shape))
    for i in range(len(data)):
        print('{}: {}'.format(i, data[i]))


def add_output_node(root, new_output_node):
    raise NotImplementedError()


def run(xml_filename, bin_filename, image_filename, output_names=None):
    net = IENetwork(model=xml_filename, weights=bin_filename)
    input_name = list(net.inputs.keys())[0]
    original_output_names = list(net.outputs.keys())

    # If we need additional output nodes, update the network description.
    additional_outputs = set(output_names) - set(original_output_names)
    if set(output_names) - set(original_output_names):
        with tempfile.NamedTemporaryFile() as temp_file:
            tree = ET.parse(xml_filename)
            for new_output in additional_outputs:
                add_output_node(tree.getroot(), new_output)
            tree.write(temp_file.name)
            net = IENetwork(model=temp_file.name, weights=bin_filename)

    if not output_names:
        output_names = original_output_names

    image_shape = net.inputs[input_name].shape
    image = PIL.Image.open(image_filename)
    image = image.resize(image_shape[2:])
    inputs = np.array(image)
    inputs = inputs.transpose((2, 0, 1))[np.newaxis, :, :, :] / 255

    ie = IECore()
    exec_net = ie.load_network(network=net, device_name='CPU')
    outputs = exec_net.infer(inputs={input_name: inputs})

    for output_name in output_names:
        dump_outputs(output_name, outputs[output_name].shape, outputs[output_name].flatten())


def main():
    parser = argparse.ArgumentParser('Run a OpenVino model')
    parser.add_argument('xml_filename', help="Filename for the OpenVino XML file")
    parser.add_argument('bin_filename', help="Filename for the OpenVino bin file")
    parser.add_argument('image_filename', help='Filename for the input image')
    parser.add_argument('--output_name', type=str, nargs='+', default=[], help='layer name to be extracted')

    args = parser.parse_args()
    run(args.xml_filename, args.bin_filename, args.image_filename, args.output_name)


if __name__ == '__main__':
    main()

# modelutils
Collection of utility scripts for ONNX/TensorFlow/CoreML/Caffe/OpenVino models.

## Install
```bash
pip install modelutils
```

If you would like to use caffe or openvino related commands, please follow their instruction to install the required packages.

## Usage
```
# Dump a model in text format.
mudump <model_filepath> [-o [<output_filepath>]]

# Serialize a model from text format.
muload <text_filepath> <output_filepath>

# Show a summary of a model.
musummarize <model_filepath> [<weights_filepath>]

# Run a model.
murun <model_filepath> [<input_npy_filepath>] [-o <output_npy_filepath>] [--output_name <node_name>]

# Get weights from a model
mugetdata <model_filepath> [--name <data_name>]

# Set weights to a model
musetdata <model_filepath> --name <data_name>

# Get diff of two npy arrays.
npdiff <input_filepath0> <input_filepath1>

# Convert an image to npy.
npimage <image_filepath> [--size <input_size>] [--scale <scale>] [--subtract <subtract>]

# Convert a npy to json.
npjson [<npy_filepath>]

# np.max() array
npmax [<npy_filepath>]

# print npy in human-friendly way
npprint [<npy_filepath>]

# np.transpose() array
nptranspose [<npy_filepath>] --perm [<perm> [<perm> ...]]

# np.zeros() array
npzeros <dim> [<dim> [<dim> ...]]
```

## Examples
```bash
# Run inference with an image file.
npimage image.jpg --size 224 | murun mobilenetv2.onnx

# Same
npimage image.jpg --size 224 > image.npy
murun mobilenetv2.onnx image.npy

# Save the result to npy
npimage image.png --size 224 | murun mobilenetv2.onnx -o result.npy

# Run inference with zero array (maybe for model debugging)
npzeros 1 224 224 3 | murun mobilenetv2.tflite

# Get intermediate layer values
npimage image.bmp --size 320 --scale 255 | murun mobilenetv2.pb --output_name conv1:0

# Compare two model outputs
npdiff <(murun model0.onnx image.npy) <(murun model1.onnx image.npy) | npmax

# Copy weights of one model to another model
mugetdata model.onnx --name conv1 | nptranspose --perm 2 3 1 0 | musetdata model.pb --name conv1 > new_model.pb
```

## Notes
There are lots of small scripts in onnx/tensorflow/coreml/caffe/openvino directories. They can be good examples when you would like to play with various models.
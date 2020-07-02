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
murun <model_filepath> [<input_npy_filepath>] [--output_name <node_name>]

# Get weights from a model
mugetdata <model_filepath> [--name <data_name>] [--type <model_type>]

# Set weights to a model
musetdata <model_filepath> --name <data_name> [--type <model_type>]

# Get diff of two npy arrays.
npdiff <input_filepath0> <input_filepath1>

# Convert an image to npy.
npimage <image_filepath> [--size <input_size>] [--scale <scale>] [--subtract <subtract>] [--bgr]

# Convert a npy to json.
npjson [<npy_filepath>]

# np.max() array
npmax [<npy_filepath>]

# print npy in human-friendly way
npprint [<npy_filepath>] [-s] [-a]

# np.transpose() array
nptranspose [<perm> [<perm> ...]]

# np.zeros() array
npzeros <dim> [<dim> [<dim> ...]]
```

## Examples
```bash
# Run inference with an image file.
npimage image.jpg --size 224 | murun mobilenetv2.pb > result.npy

# If the model is onnx and the input expects BGR [0-255],
npimage image.jpg --size 224 --scale 255 --bgr | nptranspose 0 3 1 2 | murun mobilenetv2.onnx > result.npy

# Run inference with zero array (maybe for model debugging)
npzeros 1 224 224 3 | murun mobilenetv2.tflite | npprint

# Get intermediate layer values
npimage image.bmp --size 320 --scale 255 | murun mobilenetv2.pb --output_name conv1:0 | npprint

# Compare two model outputs
npdiff <(murun model0.onnx image.npy) <(murun model1.onnx image.npy) | npmax

# Copy weights of one model to another model
mugetdata model.onnx --name conv1 | nptranspose --perm 2 3 1 0 | musetdata model.pb --name conv1 > new_model.pb
```

## Notes
There are lots of small scripts in onnx/tensorflow/coreml/caffe/openvino directories. They can be good examples when you would like to play with various models.
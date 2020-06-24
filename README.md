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

# Run a model. input_filepath can be *.{jpg|bmp|png|npy}. If not specified, read npy from stdin.
murun <model_filepath> [<input_filepath>]

# Convert an image to npy.
npimage <image_filepath> [--size <input_size>] [--scale <scale>] [--subtract <subtract>]

# np.zeros() array
npzeros dim [dim [dim ...]]
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
```

## Notes
There are lots of small scripts in onnx/tensorflow/coreml/caffe/openvino directories. They can be good examples when you would like to play with various models.
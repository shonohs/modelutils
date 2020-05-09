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
```
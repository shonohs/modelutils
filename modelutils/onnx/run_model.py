import tempfile
import pathlib
import onnx
import onnxruntime


def _add_output_node(input_model_filepath, output_model_filepath, output_node_names):
    model = onnx.ModelProto()
    model.ParseFromString(input_model_filepath.read_bytes())
    existing_output_names = set([o.name for o in model.graph.output])
    for output_name in set(output_node_names) - existing_output_names:
        model.graph.output.append(onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None))

    output_model_filepath.write_bytes(model.SerializeToString())


def run_model(model_filepath, input_array, output_names):
    if output_names:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_model_filepath = pathlib.Path(tempdir) / 'model.onnx'
            _add_output_node(model_filepath, temp_model_filepath, output_names)
            sess = onnxruntime.InferenceSession(str(temp_model_filepath))

    else:
        sess = onnxruntime.InferenceSession(str(model_filepath))
        output_names = [o.name for o in sess.get_outputs()]

    input_array = input_array.transpose((0, 3, 1, 2))
    outputs = sess.run(output_names, {sess.get_inputs()[0].name: input_array})

    return {n: outputs[i] for i, n in enumerate(output_names)}

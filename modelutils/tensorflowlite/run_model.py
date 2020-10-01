import pathlib
import sys
import numpy as np


def run_model(model_filepath, input_array, output_names):
    import tensorflow
    interpreter = tensorflow.lite.Interpreter(model_path=str(model_filepath))
    output_details = interpreter.get_output_details()

    if output_names:
        intermediate_tensors = set(output_names) - set(d['name'] for d in output_details)
        if intermediate_tensors:
            # In this case, we need to update the model to add the new output.
            indexes = [d['index'] for d in interpreter.get_tensor_details() if d['name'] in intermediate_tensors]
            sys.path.append(str(pathlib.Path(__file__).resolve().parent))
            import flatbuffers
            from tflite.Model import ModelT, Model

            model_binary = model_filepath.read_bytes()
            model = ModelT.InitFromObj(Model.GetRootAsModel(model_binary, 0))
            assert len(model.subgraphs) == 1
            model.subgraphs[0].outputs = np.append(model.subgraphs[0].outputs, indexes).astype(dtype=model.subgraphs[0].outputs.dtype)
            builder = flatbuffers.Builder(len(model_binary) + 128)
            offset = model.Pack(builder)
            builder.Finish(offset, file_identifier=b'\x54\x46\x4C\x33')

            interpreter = tensorflow.lite.Interpreter(model_content=bytes(builder.Output()))

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()

    assert len(input_details) == 1
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()

    return {d['name']: interpreter.get_tensor(d['index']) for d in interpreter.get_output_details() if not output_names or d['name'] in output_names}

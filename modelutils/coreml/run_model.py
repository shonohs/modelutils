import tempfile
import pathlib
import numpy as np
import PIL.Image


def _add_output_node(input_model_filepath, output_model_filepath, output_node_names):
    import coremltools
    model = coremltools.models.MLModel(str(input_model_filepath)).get_spec()

    for name in output_node_names:
        output = model.description.output.add()
        output.name = name
        output.type.multiArrayType.dataType = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32

    output_model_filepath.write_bytes(model.SerializeToString())


def run_model(model_filepath, input_array, output_names):
    import coremltools

    image = PIL.Image.fromarray(input_array[0].astype(np.uint8))

    if output_names:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_model_filepath = pathlib.Path(tempdir) / 'model.mlmodel'
            _add_output_node(model_filepath, temp_model_filepath, output_names)
            model = coremltools.models.MLModel(str(temp_model_filepath))
    else:
        model = coremltools.models.MLModel(str(model_filepath))
        output_names = [o.name for o in model.get_spec().description.output]

    input_names = [i.name for i in model.get_spec().description.input]
    assert len(input_names) == 1
    input_name = input_names[0]

    predictions = model.predict({input_name: image})

    return {name: predictions[name] for name in output_names}

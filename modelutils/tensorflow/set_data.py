import sys


def set_data(model_filepath, name, value):
    import tensorflow
    graph_def = tensorflow.compat.v1.GraphDef()
    graph_def.ParseFromString(model_filepath.read_bytes())

    new_tensor = tensorflow.make_tensor_proto(value)

    for node in graph_def.node:
        if node.op == 'Const' and node.name == name:
            node.attr['value'].tensor.CopyFrom(new_tensor)
            return graph_def.SerializeToString()

    print(f"Adding a new const node: {name}", file=sys.stderr)

    node = graph_def.node.add()
    node.op = 'Const'
    node.name = name
    node.attr['value'].tensor.CopyFrom(new_tensor)
    node.attr['dtype'].type = new_tensor.dtype

    return graph_def.SerializeToString()

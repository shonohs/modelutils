import tensorflow

def get_data(model_filepath, name):
    graph_def = tensorflow.compat.v1.GraphDef()
    graph_def.ParseFromString(model_filepath.read_bytes())

    for node in graph_def.node:
        if node.op == 'Const' and node.name == name:
            return tensorflow.make_ndarray(node.attr['value'].tensor)

    return None

import torch


def get_data(model_filepath, name):
    state_dict = torch.load(model_filepath, map_location=torch.device('cpu'))
    if name:
        return state_dict[name].numpy()

    return {name: state_dict[name].numpy() for name in state_dict}

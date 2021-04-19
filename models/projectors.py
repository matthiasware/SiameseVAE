import torch


def get_linear_projection_block_layers(
        d_in,
        d_out,
        dropout_rate=0.1,
        normalize=False,
        activation=True,
        bias=False):
    layers = []
    layers.append(torch.nn.Linear(d_in, d_out, bias=False))
    if normalize:
        layers.append(torch.nn.BatchNorm1d(d_out))
    if activation:
        layers.append(torch.nn.ReLU(inplace=True))
    if dropout_rate is not None:
        layers.append(torch.nn.Dropout(p=dropout_rate))
    return layers


def get_projection_head_layers(
        d_in,
        d_out,
        d_hidden,
        n_hidden,
        normalize,
        dropout_rate,
        activation_last,
        normalize_last,
        dropout_rate_last):

    dims = [d_in]
    for _ in range(n_hidden):
        dims.append(d_hidden)
    dims.append(d_out)
    #
    all_layers = []
    for i in range(len(dims) - 2):
        layers = get_linear_projection_block_layers(
            dims[i], dims[i + 1], dropout_rate, normalize)
        all_layers += layers

    all_layers += get_linear_projection_block_layers(
        dims[-2], dims[-1], dropout_rate_last, normalize_last, activation_last)
    return all_layers


def get_projector(**kwargs):
    layers = get_projection_head_layers(**kwargs)
    projector = torch.nn.Sequential(*layers)
    return projector
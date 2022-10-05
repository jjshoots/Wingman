import torch


def gpuize(input, device):
    """gpuize.

    Args:
        input: the array that we want to gpuize
        device: a string of the device we want to move the thing to
    """
    if torch.is_tensor(input):
        if input.device == device:
            return input.float()
        return input.to(device).float()
    return torch.tensor(input).float().to(device)


def cpuize(input):
    """cpuize.

    Args:
        input: the array of the thing we want to put on the cpu
    """
    if torch.is_tensor(input):
        return input.detach().cpu().numpy()
    else:
        return input


def shutdown_handler(*_):
    """shutdown_handler.

    Args:
        _:
    """
    print("ctrl-c invoked")
    exit(0)

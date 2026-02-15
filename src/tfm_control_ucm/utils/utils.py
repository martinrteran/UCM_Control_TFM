import torch
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    try:
        torch.Tensor([1, 2], device=device)
    except RuntimeError:
        device = torch.device("cpu")
    return device

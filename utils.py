import string

import torch

vocab = string.ascii_lowercase + ' .'


def one_hot(s: str):
    import numpy as np
    if len(s) == 0:
        return torch.zeros((len(vocab), 0))
    return torch.as_tensor(np.array(list(s.lower()))[None, :] == np.array(list(vocab))[:, None]).float()

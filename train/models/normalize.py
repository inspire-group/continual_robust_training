import torch
import torch.nn as nn

class Normalize(nn.Module):
    '''A layer for normalizing input before passing it into the network'''
    def __init__(self, mean, std, base_model):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.base_model = base_model

    def forward(self, x, **kwargs):
        normalized = (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]
        return self.base_model(x, **kwargs)

def apply_normalization(base_model, mean, std):
    return Normalize(mean, std, base_model)

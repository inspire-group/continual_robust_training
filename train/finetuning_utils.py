
# TODO: add code for changing certain layers to be not learnable
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

class EWC(object):
    def __init__(self, model, dataset, attacks, obj='acc', use_fisher=True):

        self.model = model
        self.dataset = dataset
        self.attacks = attacks

        self.params = {n: p for n, p in self.model.named_parameters() if (p.requires_grad and 
                                                                        "head_proj" not in n and
                                                                        "head_pred" not in n)}
        self._means = {}
        self.use_fisher = use_fisher
        self.obj = obj

        self._precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self._precision_matrices[n] = p.data.cuda()
        if self.use_fisher:
            self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = p.data.cuda()

    def _diag_fisher(self):
        self.model.eval()
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data.cuda()

        for attack in self.attacks:
            for (input, y) in self.dataset:
                self.model.zero_grad()
                input = input.cuda()
                #y = y.cuda()
                adv_input = attack(input, y)
                output = self.model(adv_input)
                label = output.max(1)[1].view(-1)
                loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and "head_proj" not in n and "head_pred" not in n:
                        precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                if self.use_fisher:
                    _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                else:
                    _loss = (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

def count_layers(model):
    count = 0
    for name, _ in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            count += 1
        elif 'fc' in name and 'weight' in name:
            count += 1
    return count


def fix_end_layers(model, index):
    cur_count = 0
    last_prefix = None
    for name, param in model.named_parameters():
        if cur_count == index:
            # check that it is not the bias of the prev layer
            if name != last_prefix + 'bias':
                param.requires_grad = True
        elif cur_count > index:
            # fix the parameter
            param.requires_grad = False
        if 'conv' in name and 'weight' in name:
            last_prefix = name.replace('weight', '')
            cur_count += 1
        elif 'fc' in name and 'weight' in name:
            last_prefix = name.replace('weight', '')
            cur_count += 1
    


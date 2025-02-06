import yaml
import json
from os.path import exists
import numpy as np
import torch

def bin_search(model, batch, target, atks, low, high, correct_per_img):
    is_done = low > high
    cur_atks = {}
    cur_batch = []
    cur_targets = []
    new_low = np.zeros(len(low))
    new_high = np.zeros(len(low))
    if all(is_done):
        return correct_per_img

    for i in range(len(is_done)):
        if is_done[i]:
            new_low[i] = low[i]
            new_high[i] = high[i]
            continue
        else:
            mid = int((low[i] + high[i])/2)
            if mid not in cur_atks:
                cur_atks[mid] = [i]
            else:
                cur_atks[mid].append(i)
    
    # generate adversarial examples
    for atk_idx in cur_atks:
        atk_batch = batch[cur_atks[atk_idx]].cuda()
        atk_targets = target[cur_atks[atk_idx]].cuda()
        atk = atks[atk_idx]
        cur_batch.append(atk(atk_batch, atk_targets))
        cur_targets.append(atk_targets)
    cur_batch = torch.cat(cur_batch, dim=0)
    cur_targets = torch.cat(cur_targets).cuda()

    # check classification of batch of adversarial examples
    logits = model(cur_batch)
    correct = (logits.argmax(1) == cur_targets)

    # if correct, then we know all smaller radii also correct
    # if incorrect, then we know that all larger radii are also incorrect
    i = 0
    for atk_idx in cur_atks:
        data_inds = cur_atks[atk_idx]
        for idx in data_inds:
            is_correct = correct[i]
            i += 1
            if is_correct:
                new_low[idx] = atk_idx + 1
                new_high[idx] = high[idx]
                correct_per_img[idx][:atk_idx + 1] = 1
            else:
                new_low[idx] = low[idx]
                new_high[idx] = atk_idx - 1
                correct_per_img[idx][atk_idx:] = 0
    
    return bin_search(model, batch, target, atks, new_low, new_high, correct_per_img)


def get_acc_bin_search(testloader, model, atks, eps, prec=5):
    '''computes the accuracy of the model with respect to each of the provided adversaries'''
    correct = np.zeros((len(testloader.dataset), len(atks)))
    img_idx = 0
    for i, (batch, target) in enumerate(testloader):
        batch = batch.cuda()
        target = target.cuda()
        correct_per_img = np.empty((len(batch), len(eps)))
        low = np.zeros(len(batch))
        high = np.ones(len(batch)) * (len(atks) - 1)
        correct_per_img = bin_search(model, batch, target, atks, low, high, correct_per_img)
        correct[img_idx:img_idx + len(batch)] = correct_per_img
        img_idx += len(batch)
    # convert correct matrix into accuracies across each perturbation type
    accs_per_eps = correct.sum(axis=0) / len(testloader.dataset) * 100
    #print(correct.shape)
    #print(accs_per_eps.shape)
    accs = {}
    for i, e in enumerate(eps):
        accs[float(round(e, prec))] = float(accs_per_eps[i])
    return accs


def get_acc(testloader, model, atks, eps, prec=5):
    '''computes the accuracy of the model with respect to each of the provided adversaries'''
    accs = {}
    dataset_len = len(testloader.dataset)
    for i in range(len(eps)):
        accs[float(round(eps[i], prec))] = 0
    for batch, target in testloader:
        batch = batch.cuda()
        target = target.cuda()
        for i, atk in enumerate(atks):
            pred = model(atk(batch, target))
            accs[round(eps[i], prec)] += (pred.argmax(1) == target).sum().item() / dataset_len * 100
    return accs

def get_acc_single(testloader, model, atk):
    '''computes the accuracy of the model with respect to each of the provided adversaries'''
    accs = 0
    dataset_len = len(testloader.dataset)
    for batch, target in testloader:
        batch = batch.cuda()
        target = target.cuda()
        pred = model(atk(batch, target))
        accs += (pred.argmax(1) == target).sum().item() / dataset_len * 100
    return accs

def load(file_name):
    if not exists(file_name):
        raise Exception('{}}is incorrect, file does not exist'.format(file_name))
    with open(file_name, 'r') as f:
        loaded = yaml.safe_load(f)
    return loaded

def save(data, out_file):
    # save data dictionary to out_file
    # if out_file already exists and contains data values for
    # attacks/epsilon values not in the data dictionary, we will copy those attacks
    # over to the data dictionary and save the aggregate data dict
    if exists(out_file):
        with open(out_file, 'r') as f:
            data_curr = yaml.load(f)
        for k in data_curr:
            if k not in data:
                data[k] = data_curr[k]
            elif type(data_curr[k]) is dict:
                for j in data_curr[k]:
                    if j not in data[k]:
                        data[k][j] = data_curr[k][j]
    with open(out_file + '.yml', 'w') as f:
        documents = yaml.dump(data, f)
    with open(out_file + '.json', 'w') as f:
        documents2 = json.dump(data, f)

def save_json(data, outfile):
    with open(outfile, 'w') as f:
        documents = json.dump(data, f)

def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'classifier' in k:
            k = k.replace('classifier', 'fc')
        if 'linear' in k:
            k = k.replace('linear', 'fc')
        if 'shortcut' in k:
            k = k.replace('shortcut', 'downsample')
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

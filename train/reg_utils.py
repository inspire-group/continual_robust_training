from __future__ import print_function
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from reg_utils_non_lp import StAdvReg, ReColorAdvReg, LPIPSReg
import random
import attacks.uar_attacks._regularizers as uar_regularizers

class Regularizer(object):
    def __init__(self, model, regtype, attack, epsilon, random, uniform_freq=0, epsilon_noise_only=0, include_orig=True, temp=0.5, loss_params=1, num_steps=10, step_size=0.01, dataset_name="cifar"):
        if random != 'adv': 
            # using random noise
            self.loss = NoiseReg(model, epsilon, random)
        elif attack == 'LinfAttack' or attack == 'L2Attack' or random != 'adv':
            if regtype == "var_reg":
                self.loss = L2Reg(model, epsilon, attack, num_steps, step_size, var_reg=True)
            elif regtype == "l2":
                self.loss = L2Reg(model, epsilon, attack, num_steps, step_size, var_reg=False)
            elif regtype == "contrastive" or regtype == 'asym_contrastive':
                self.loss = ContrastiveLoss(model, attack, epsilon, include_orig, temp, loss_params, num_steps)
        # currently worst-case constrastive losses regularization not supported for non Lp attacks
        elif attack == 'StAdvAttack':
            self.loss = StAdvReg(model, bound=epsilon, variation_reg = (regtype=='var_reg'), num_iterations=num_steps, step_size=step_size)
        elif attack == 'ReColorAdvAttack':
            self.loss = ReColorAdvReg(model, bound=epsilon, variation_reg = (regtype=='var_reg'), num_iterations=num_steps, step_size=step_size)
        elif attack == 'FastLagrangePerceptualAttack':
            if dataset_name == "cifar":
                self.loss = LPIPSReg(model, bound=epsilon, lpips_model='alexnet_cifar', projection='newtons', variation_reg = (regtype=='var_reg'), num_iterations=num_steps)
            else:
                self.loss = LPIPSReg(model, bound=epsilon, lpips_model='alexnet', projection='newtons', variation_reg = (regtype=='var_reg'), num_iterations=num_steps)
        else:
            reg_name = attack[:-6] + "Reg" # remove "Attack" from name and replace with Reg
            self.loss = uar_regularizers.__dict__[reg_name](model, task=regtype, dataset_name=dataset_name, bound=epsilon, num_iterations=num_steps, step_size=step_size)
            
        self.random = random
        self.uniform_freq = uniform_freq
        if self.uniform_freq > 0:
            self.loss_uniform = NoiseReg(model, epsilon_noise_only, 'uniform')

    def compute_loss(self, x, images_t1=None, images_t2=None):
        if self.uniform_freq > 0:
            if random.random() < self.uniform_freq:
                if images_t1 is not None:
                    return self.loss_uniform(x, images_t1=images_t1, images_t2=images_t2)
                return self.loss_uniform(x)
        else:
            if images_t1 is not None:
                return self.loss(x, images_t1=images_t1, images_t2=images_t2)
            return self.loss(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, model, attack, epsilon, include_orig, temp, loss_params, num_steps, step_size=0.01):
        super(ContrastiveLoss, self).__init__()
        self.model = model
        self.attack = attack
        self.epsilon = epsilon
        self.include_orig = include_orig
        self.num_steps = num_steps
        self.step_size = step_size
        _, _, _, self.use_self_features = loss_params
        if criterion == 'contrastive':
            self.crit = ori_SupConLoss(loss_params, temperature=temp)
        else: # use asymmetric NCE
            self.crit = SupConLoss(loss_params, temperature=temp)
    
    def forward(self, x, images_t1=None, images_t2=None):
        x_new = get_max_for_contrastive(x, self.model, self.crit, self.epsilon, self.attack, self.step_size, True, self.num_steps, 0, 1, None, images_t1, images_t2, self.include_orig, self.use_self_features)
        return compute_contrastive_loss(x_new, x, self.model, self.crit, images_t1=images_t1, images_t2=images_t2, include_orig=self.include_orig, use_self_features=self.use_self_features)

class NoiseReg(nn.Module):
    def __init__(self, model, epsilon, random):
        super(NoiseReg, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.random = random
    
    def forward(self, x):
        if self.random == "gaussian":
            eta = self.epsilon * torch.randn_like(x).cuda()
            x_new = torch.clamp(x + eta, 0, 1).detach()
        else:
            # it is uniform reg
            eta = torch.FloatTensor(x.shape).uniform_(-1, 1).cuda()
            eta.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_new = torch.clamp(x + eta, 0, 1).detach()
        return torch.norm(self.model(x_new) - self.model(x), 2, dim=-1).mean()


class L2Reg(nn.Module):
    def __init__(self, model, epsilon, norm, num_steps, step_size, var_reg=False, var_norm='l_2'):
        super(L2Reg, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.num_steps = num_steps
        self.step_size = step_size
        self.var_norm = var_norm
        self.var_reg = var_reg

    def forward(self, x):
        with torch.no_grad():
            x1, x2 = self.get_max(x)
        return torch.norm(self.model(x1) - self.model(x2), 2, dim=-1).mean()

    def get_max(self, x, is_random=True):
        if self.norm == 'LinfAttack':
            if is_random:
                random_noise_1 = (
                    torch.FloatTensor(x.shape)
                    .uniform_(-self.epsilon, self.epsilon)
                    .cuda()
                    .detach()
                )
            x_pgd = x + random_noise_1
            if self.var_reg:
                random_noise_2 = (
                    torch.FloatTensor(x.shape)
                    .uniform_(-self.epsilon, self.epsilon)
                    .cuda()
                    .detach()
                )
                x_pgd2 = x + random_noise_2
            for _ in range(self.num_steps):
                x_pgd.requires_grad=True
                if self.var_reg:
                    x_pgd2.requires_grad = True
                with torch.enable_grad():
                    if self.var_reg:
                        loss = torch.norm(self.model(x_pgd)- self.model(x_pgd2),
                                float(self.var_norm.split('_')[1]), dim=-1).mean()
                    else:
                        loss = torch.norm(self.model(x_pgd)- self.model(x),
                                float(self.var_norm.split('_')[1]), dim=-1).mean()
                    loss.backward()
                x_pgd = x_pgd + self.step_size * x_pgd.grad.sign()
                eta = torch.clamp(x_pgd - x, -self.epsilon, self.epsilon)
                x_pgd = torch.clamp(x + eta, 0, 1).detach()
                if self.var_reg:
                    x_pgd2 = x_pgd2 + self.step_size * x_pgd2.grad.sign()
                    eta = torch.clamp(x_pgd2 - x, -self.epsilon, self.epsilon)
                    x_pgd2 = torch.clamp(x + eta, 0, 1).detach()
            if self.var_reg:
                return x_pgd, x_pgd2
            return x_pgd, x

        if self.norm == 'L2Attack':
            if is_random:
                random_noise_1 = (
                    torch.FloatTensor(x.shape).uniform_(-1, 1).cuda().detach()
                )
                random_noise_1.renorm_(p=2, dim=0, maxnorm=self.epsilon)
            x_pgd = x + random_noise_1
            if self.var_reg:
                random_noise_2 = (
                    torch.FloatTensor(x.shape).uniform_(-1, 1).cuda().detach()
                )
                random_noise_2.renorm_(p=2, dim=0, maxnorm=self.epsilon)
                x_pgd2 = x + random_noise_2
            for _ in range(self.num_steps):
                x_pgd.requires_grad=True
                if self.var_reg:
                    x_pgd2.requires_grad=True
                with torch.enable_grad():
                    if self.var_reg:
                        loss = torch.norm(self.model(x_pgd)- self.model(x_pgd2),
                                float(self.var_norm.split('_')[1]), dim=-1).mean()
                    else:
                        loss = torch.norm(self.model(x_pgd)- self.model(x),
                                float(self.var_norm.split('_')[1]), dim=-1).mean()
                    loss.backward()
                    grad1 = x_pgd.grad.detach()
                    if self.var_reg:
                        grad2 = x_pgd2.grad.detach()
                
                # renorming gradient
                grad_norms = grad1.view(len(x), -1).norm(p=2, dim=1)
                grad1.div_(grad_norms.view(-1, 1, 1, 1))
                
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    grad1[grad_norms == 0] = torch.randn_like(
                        grad1[grad_norms == 0]
                    )
                
                x_pgd = x_pgd + self.step_size * grad1
                eta = x_pgd - x
                eta.renorm_(p=2, dim=0, maxnorm=self.epsilon)
                x_pgd = torch.clamp(x.data + eta, 0, 1).detach()

                if self.var_reg:
                    # renorming gradient
                    grad_norms = grad2.view(len(x), -1).norm(p=2, dim=1)
                    grad2.div_(grad_norms.view(-1, 1, 1, 1))
                    
                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any():
                        grad2[grad_norms == 0] = torch.randn_like(
                            grad2[grad_norms == 0]
                        )
                    
                    x_pgd2 = x_pgd2 + self.step_size * grad2
                    eta = x_pgd2 - x
                    eta.renorm_(p=2, dim=0, maxnorm=self.epsilon)
                    x_pgd2 = torch.clamp(x.data + eta, 0, 1).detach()

            if self.var_reg:
                return x_pgd, x_pgd2
            return x_pgd, x
    
def compute_contrastive_loss(perturb, images_org, model, criterion, images_t1=None, images_t2=None, include_orig=True, use_self_features=False):
    f_proj, _ = model(perturb, contrast=True)
    f_orig_proj, _ = model(images_org, contrast=True, use_self_features = use_self_features)
    if images_t1 is not None:
        f1_proj, _ = model(images_t1, contrast=True, use_self_features = use_self_features)
    if images_t2 is not None:
        f2_proj, _ = model(images_t2, contrast=True, use_self_features = use_self_features)
    if images_t1 is not None and images_t2 is not None:
        if include_orig:
            features = torch.cat(
                [f_proj.unsqueeze(1), f_orig_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
        else:
            features = torch.cat(
                [f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
    elif images_t1 is not None:
        if include_orig:
            features = torch.cat(
                [f_proj.unsqueeze(1), f_orig_proj.unsqueeze(1), f1_proj.unsqueeze(1)], dim=1)
        else:
            features = torch.cat(
                [f_proj.unsqueeze(1), f1_proj.unsqueeze(1)], dim=1)
    else:
        features = torch.cat(
                [f_proj.unsqueeze(1), f_orig_proj.unsqueeze(1)], dim=1)
    
    loss = criterion(features, stop_grad=False)
    return loss

def get_max_for_contrastive(images_org, model, criterion, epsilon, norm, step_size=0.01, is_random=True, num_steps=10, clip_min=0, clip_max=1, normalize=None, images_t1=None, images_t2=None, include_orig=True, use_self_features=False):
    x_cl = images_org.clone().detach().cuda()
    if is_random:
            if norm == 'LinfAttack':
                x_cl = x_cl + torch.zeros_like(x_cl).uniform_(-epsilon, epsilon).cuda()
            elif norm == 'L2Attack':
                random_noise = torch.zeros_like(x_cl).uniform_(-1, 1).cuda()
                random_noise.renorm_(p=2, dim=0, maxnorm=epsilon)
                x_cl = x_cl + random_noise
    x_cl.requires_grad = True 
    for _ in range(num_steps):
        f_proj, _ = model(x_cl, contrast=True, use_self_features = use_self_features)
        f_orig_proj, _ = model(images_org, contrast=True, use_self_features = use_self_features)
        if images_t1 is not None:
            f1_proj, _ = model(images_t1, contrast=True, use_self_features = use_self_features)
        if images_t2 is not None:
            f2_proj, _ = model(images_t2, contrast=True, use_self_features = use_self_features)
        if images_t1 is not None and images_t2 is not None:
            if include_orig:
                features = torch.cat(
                    [f_proj.unsqueeze(1), f_orig_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
            else:
                features = torch.cat(
                    [f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
        elif images_t1 is not None:
            if include_orig:
                features = torch.cat(
                    [f_proj.unsqueeze(1), f_orig_proj.unsqueeze(1), f1_proj.unsqueeze(1)], dim=1)
            else:
                features = torch.cat(
                    [f_proj.unsqueeze(1), f1_proj.unsqueeze(1)], dim=1)
        else:
            features = torch.cat(
                    [f_proj.unsqueeze(1), f_orig_proj.unsqueeze(1)], dim=1)
        
        loss = criterion(features, stop_grad=False)
        loss.backward()
        grad1 = x_cl.grad.detach()

        if norm == 'LinfAttack':
            x_cl.data = x_cl.data + step_size * grad1.data.sign()
            eta = torch.clamp(x_cl.data - images_org.data, -epsilon, epsilon)
            x_cl.data = torch.clamp(images_org.data + eta, clip_min, clip_max)
            return x_cl
        
        elif norm == 'L2Attack':
            grad_norms = grad1.view(len(images_org), -1).norm(p=2, dim=1)
            
            grad1.div_(grad_norms.view(-1, 1, 1, 1))
            
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                grad1[grad_norms == 0] = torch.randn_like(
                    grad1[grad_norms == 0]
                )
            
            x_cl.data += step_size * grad1.data
            eta = x_cl.data - images_org.data
            eta.renorm_(p=2, dim=0, maxnorm=epsilon)
            x_cl.data = torch.clamp(images_org.data + eta, clip_min, clip_max) 
            return x_cl         

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: XXXX.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, params, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.tau_plus, self.beta, self.adv_weight, _ = params

    def forward(self, features, labels=None, mask=None, stop_grad=False, stop_grad_sd=-1.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        XXXX
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        if stop_grad:
            anchor_dot_contrast_stpg = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T.detach()),
                self.temperature)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # For hard negatives, code adapted from HCL (XXXX)
        # =============== hard neg params =================
        tau_plus = self.tau_plus
        beta = self.beta
        temperature = 0.5
        N = (batch_size - 1) * contrast_count
        # =============== reweight neg =================
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        exp_logits_neg = exp_logits * (1 - mask) * logits_mask
        exp_logits_pos = exp_logits * mask
        pos = exp_logits_pos.sum(dim=1) / mask.sum(1)

        imp = (beta * (exp_logits_neg + 1e-9).log()).exp()
        reweight_logits_neg = (imp * exp_logits_neg) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_logits_neg.sum(dim=-1)) / (1 - tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
        log_prob = -torch.log(exp_logits / (pos + Ng))
        # ===============================================

        loss_square = mask * log_prob  # only positive positions have elements

        # mix_square = exp_logits
        mix_square = loss_square

        if stop_grad:
            logits_max_stpg, _ = torch.max(anchor_dot_contrast_stpg, dim=1, keepdim=True)
            logits_stpg = anchor_dot_contrast_stpg - logits_max_stpg.detach()
            # =============== reweight neg =================
            exp_logits_stpg = torch.exp(logits_stpg)
            exp_logits_neg_stpg = exp_logits_stpg * (1 - mask) * logits_mask
            exp_logits_pos_stpg = exp_logits_stpg * mask
            pos_stpg = exp_logits_pos_stpg.sum(dim=1) / mask.sum(1)

            imp_stpg = (beta * (exp_logits_neg_stpg + 1e-9).log()).exp()
            reweight_logits_neg_stpg = (imp_stpg * exp_logits_neg_stpg) / imp_stpg.mean(dim=-1)
            Ng_stpg = ((-tau_plus * N * pos_stpg + reweight_logits_neg_stpg.sum(dim=-1)) / (1 - tau_plus))

            # constrain (optional)
            Ng_stpg = torch.clamp(Ng_stpg, min=N * np.e ** (-1 / temperature))
            log_prob_stpg = -torch.log(exp_logits_stpg / (pos_stpg + Ng_stpg))
            # ===============================================
            tmp_square = mask * log_prob_stpg
        else:
            # tmp_square = exp_logits
            tmp_square = loss_square
        if stop_grad:
            ac_square = stop_grad_sd * tmp_square[batch_size:, 0:batch_size].T + (1 - stop_grad_sd) * tmp_square[
                                                                                                      0:batch_size,
                                                                                                      batch_size:]
        else:
            ac_square = tmp_square[0:batch_size, batch_size:]

        mix_square[0:batch_size, batch_size:] = ac_square * self.adv_weight
        mix_square[batch_size:, 0:batch_size] = ac_square.T * self.adv_weight

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = mix_square.sum(1) / mask.sum(1)

        # loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ori_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: XXXX.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, params, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(ori_SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        _, _, self.adv_weight, _ = params

    def forward(self, features, labels=None, mask=None, stop_grad=False, stop_grad_sd=-1.0):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        XXXX
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.size(0), features.size(1), -1)

        batch_size = features.size(0)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        if stop_grad:
            anchor_dot_contrast_stpg = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T.detach()),
                self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        loss_square = mask * log_prob  # only positive position have elements
        mix_square = loss_square
        if stop_grad:
            logits_max_stpg, _ = torch.max(anchor_dot_contrast_stpg, dim=1, keepdim=True)
            logits_stpg = anchor_dot_contrast_stpg - logits_max_stpg.detach()
            # compute log_prob
            exp_logits_stpg = torch.exp(logits_stpg) * logits_mask
            log_prob_stpg = logits_stpg - torch.log(exp_logits_stpg.sum(1, keepdim=True))
            loss_square_stpg = mask * log_prob_stpg
            tmp_square = loss_square_stpg
        else:
            tmp_square = loss_square
        if stop_grad:
            ac_square = stop_grad_sd * tmp_square[batch_size:, 0:batch_size].T + (1 - stop_grad_sd) * tmp_square[
                                                                                                      0:batch_size,
                                                                                                      batch_size:]
        else:
            ac_square = tmp_square[0:batch_size, batch_size:]
        #print(batch_size)
        #print(mix_square.shape)
        #print(self.adv_weight)
        mix_square[0:batch_size, batch_size:] = ac_square * self.adv_weight
        mix_square[batch_size:, 0:batch_size] = ac_square.T * self.adv_weight

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = mix_square.sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

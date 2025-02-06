import torch
import torch.nn as nn
from perceptual_advex.attacks import MisterEdAttack, UARAttack
from perceptual_advex.perceptual_attacks import LagrangePerceptualAttack, PerceptualPGDAttack
import autoattack
from autoattack.autopgd_base import APGDAttack
import kornia
import torch.nn.functional as F
from numpy import pi

# mister_ed
from recoloradv.mister_ed import loss_functions as lf
from recoloradv.mister_ed import adversarial_training as advtrain
from recoloradv.mister_ed import adversarial_perturbations as ap 
from recoloradv.mister_ed import adversarial_attacks as aa
from recoloradv.mister_ed import spatial_transformers as st

# ReColorAdv
from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "Linf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "L2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "Linf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "L2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

class LpPGDAttack(nn.Module):
    def __init__(self, model, attack_norm, bound,
                 num_iterations=10, step=None, restarts=1, early_stop=False):
        super().__init__()

        if step is None:
            step = bound / (num_iterations - 2)
        self.model = model
        self.attack_norm = attack_norm
        self.bound = bound
        self.alpha = step
        self.num_iterations = num_iterations
        self.restarts = restarts
        self.early_stop = early_stop

    def forward(self, inputs, labels):
        perturbation = attack_pgd(self.model, inputs, labels, self.bound, self.alpha, self.num_iterations, self.restarts,
               self.attack_norm, self.early_stop)
        return inputs + perturbation

# PGD
class LinfAttack(LpPGDAttack):
    def __init__(self, model, bound=8/255, **kwargs):
        if 'num_iterations' in kwargs:
            step = bound / (kwargs['num_iterations'] / 10 * 4)
        else:
            step = bound / 4
        super().__init__(
            model,
            attack_norm='Linf',
            bound=bound,
            step=step,
            **kwargs,
        )


class L2Attack(LpPGDAttack):
    def __init__(self, model, bound=1, **kwargs):
        if 'num_iterations' in kwargs:
            step = bound / (kwargs['num_iterations'] / 10 * (0.5 / 0.075))
        else:
            step = bound / (0.5 / 0.075)
        super().__init__(
            model,
            attack_norm='L2',
            bound=bound,
            step=step,
            **kwargs,
        )

# use L1 APGD from XXXX.pdf for training
class L1Attack(nn.Module):
    def __init__(self, model, bound=12, **kwargs):
        super().__init__()

        self.model = model
        self.attack = APGDAttack(model, norm='L1', eps=bound)

    def forward(self, inputs, labels):
        return self.attack.perturb(inputs, y=labels)

# APGD Linf, L2, L1
class AutoAttack(nn.Module):
    def __init__(self, model, full=False, **kwargs):
        super().__init__()

        kwargs.setdefault('verbose', False)
        self.model = model
        self.kwargs = kwargs
        self.attack = None
        self.full = full
        #print(self.full)

    def forward(self, inputs, labels):
        # Necessary to initialize attack here because for parallelization
        # across multiple GPUs.
        if self.attack is None:
            self.attack = autoattack.AutoAttack(
                self.model, device=inputs.device, **self.kwargs)
            if not self.full:
                self.attack.attacks_to_run = ['apgd-t', 'fab-t']
            print(self.attack.attacks_to_run) 
        return self.attack.run_standard_evaluation(inputs, labels)



class AutoLinfAttack(AutoAttack):
    def __init__(self, model, bound=8/255, full=False, **kwargs):
        super().__init__(
            model,
            norm='Linf',
            eps=bound,
            full=full
        )


class AutoL2Attack(AutoAttack):
    def __init__(self, model, bound=1, full=False, **kwargs):
        super().__init__(
            model,
            norm='L2',
            eps=bound,
            full=full
        )

class AutoL1Attack(AutoAttack):
    def __init__(self, model, bound=12, full=False, **kwargs):
        super().__init__(
            model,
            norm='L1',
            eps=bound,
            full=full
        )


class StAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.05, num_iterations=20, **kwargs):
        lr = bound/ (num_iterations - 2)
        super().__init__(
            model,
            lr = lr,
            threat_model=lambda: ap.ThreatModel(ap.ParameterizedXformAdv, {
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_class': st.FullSpatial,
                'use_stadv': True,
            }),
            perturbation_norm_loss=0.0025 / bound,
        )


class ReColorAdvAttack(MisterEdAttack):
    def __init__(self, model, bound=0.06, num_iterations=20, **kwargs):
        lr = bound/ (num_iterations - 2)
        super().__init__(
            model,
            lr = lr,
            threat_model=lambda: ap.ThreatModel(pt.ReColorAdv, {
                'xform_class': ct.FullSpatial,
                'cspace': cs.CIELUVColorSpace(),
                'lp_style': 'inf',
                'lp_bound': bound,
                'xform_params': {
                  'resolution_x': 16,
                  'resolution_y': 32,
                  'resolution_z': 32,
                },
                'use_smooth_loss': True,
            }),
            perturbation_norm_loss=0.0036 / bound,
        )


class LPIPSAttack(nn.Module):
    def __init__(self, model, bound=0.5, num_iterations=40, lpips_model='alexnet_cifar'):
        super().__init__()
        self.model = model
        self.ppgd = PerceptualPGDAttack(model, bound=bound, num_iterations=num_iterations, lpips_model=lpips_model)
        self.lpa = LagrangePerceptualAttack(model, bound=bound, num_iterations=num_iterations, lpips_model=lpips_model)

    def forward(self, inputs, labels):
        ppgd_adv = self.ppgd(inputs, labels)
        lpa_adv = self.lpa(inputs, labels)

        correct_ppgd = (self.model(ppgd_adv).argmax(-1) == labels)
        # if ppgd is correct then use LPA
        ppgd_adv[correct_ppgd] = lpa_adv[correct_ppgd]
        return ppgd_adv

class Hue(nn.Module):
    def __init__(self, model, bound, num_iterations=20, step_size = None):
        super().__init__()
        self.model = model
        self.bound = bound # bound should be between (0, pi]
        self.num_iterations = num_iterations 
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 2 * bound / (num_iterations - 2)
    
    def forward(self, inputs, labels):
        hue = (torch.rand(labels.size()).cuda()-0.5) * 2 * self.bound
        hue.requires_grad_()
        new_data = kornia.enhance.adjust_hue(inputs, hue)
        for i in range(self.num_iterations):

            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.num_iterations:
                hue_grad = torch.autograd.grad(cost, hue, retain_graph=True)[0]
            else:
                hue_grad = torch.autograd.grad(cost, hue, retain_graph=False)[0]
            hue = torch.clamp(hue + torch.sign(hue_grad) * self.step_size, -self.bound,
                              self.bound).detach().requires_grad_()
            new_data = kornia.enhance.adjust_hue(inputs, hue)
        return new_data


class Saturation(nn.Module):
    def __init__(self, model, bound, num_iterations=20, step_size = None):
        super().__init__()
        self.model = model
        self.bound = bound # bound should be between (0, 1]
        self.num_iterations = num_iterations 
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 2 * bound / (num_iterations - 2)
    
    def forward(self, inputs, labels):
        upper = 1 + self.bound
        lower = 1 - self.bound   
        sat = (2 * self.bound * torch.rand(labels.size()).cuda()) + lower
        sat.requires_grad_()
        new_data = kornia.enhance.adjust_saturation(inputs, sat)
        for i in range(self.num_iterations):
            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.num_iterations:
                sat_grad = torch.autograd.grad(cost, sat, retain_graph=True)[0]
            else:
                sat_grad = torch.autograd.grad(cost, sat, retain_graph=False)[0]
            sat = torch.clamp(sat + torch.sign(sat_grad) * self.step_size, lower,
                              upper).detach().requires_grad_()
            new_data = kornia.enhance.adjust_saturation(inputs, sat)
        return new_data

class Brightness(nn.Module):
    def __init__(self, model, bound, num_iterations=20, step_size = None):
        super().__init__()
        self.model = model
        self.bound = bound # bound should be between (0, 1]
        self.num_iterations = num_iterations 
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 2 * bound / (num_iterations - 2)
    
    def forward(self, inputs, labels):
        brightness = torch.rand(labels.size()).cuda() * self.bound
        brightness.requires_grad_()
        new_data = kornia.enhance.adjust_brightness(inputs, brightness)
        for i in range(self.num_iterations):
            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.num_iterations:
                brightness_grad = torch.autograd.grad(cost, brightness, retain_graph=True)[0]
            else:
                brightness_grad = torch.autograd.grad(cost, brightness, retain_graph=False)[0]
            brightness = torch.clamp(brightness + torch.sign(brightness_grad) * self.step_size, 0,
                              self.bound).detach().requires_grad_()
            new_data = kornia.enhance.adjust_brightness(inputs, brightness)
        return new_data

class Contrast(nn.Module):
    def __init__(self, model, bound, num_iterations=20, step_size = None):
        super().__init__()
        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations 
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 2 * bound / (num_iterations - 2)
    
    def forward(self, inputs, labels):
        upper = 1 + self.bound
        lower = 1 - self.bound   
        contrast = (2 * self.bound * torch.rand(labels.size()).cuda()) + lower
        contrast.requires_grad_()
        new_data = kornia.enhance.adjust_contrast(inputs, contrast)
        for i in range(self.num_iterations):
            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.num_iterations:
                contrast_grad = torch.autograd.grad(cost, contrast, retain_graph=True)[0]
            else:
                contrast_grad = torch.autograd.grad(cost, contrast, retain_graph=False)[0]
            contrast = torch.clamp(contrast + torch.sign(contrast_grad) * self.step_size, lower,
                              upper).detach().requires_grad_()
            new_data = kornia.enhance.adjust_contrast(inputs, contrast)
        return new_data

class AffineWarp(nn.Module):
    def __init__(self, model, bound, num_iterations=20, step_size = None):
        super().__init__()
        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations 
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 2 * bound / (num_iterations - 2)
            
        identity = torch.eye(2,3).cuda()
        self.identity = identity.unsqueeze(0)
    
    def forward(self, inputs, labels):
        warp = torch.rand((len(labels), 2, 3)).cuda() * self.bound
        warp.requires_grad_()
        #print('input size')
        #print(inputs.size(2), inputs.size(3))
        new_data = kornia.geometry.transform.warp_affine(inputs, warp + self.identity, (inputs.size(2), inputs.size(3)))
        #print('data size')
        #print(new_data.size(2), new_data.size(3))
        for i in range(self.num_iterations):
            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.num_iterations:
                warp_grad = torch.autograd.grad(cost, warp, retain_graph=True)[0]
            else:
                warp_grad = torch.autograd.grad(cost, warp, retain_graph=False)[0]
            warp = torch.clamp(warp + torch.sign(warp_grad) * self.step_size, -self.bound,
                              self.bound).detach().requires_grad_()
            new_data = kornia.geometry.transform.warp_affine(inputs, warp + self.identity, (inputs.size(2), inputs.size(3)))
        #print(new_data.size(2), new_data.size(3))
        return new_data

class PerspectiveWarp(nn.Module):
    def __init__(self, model, bound, num_iterations=20, step_size = None):
        super().__init__()
        self.model = model
        self.bound = bound
        self.num_iterations = num_iterations 
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 2 * bound / (num_iterations - 2)
            
        identity = torch.eye(3).cuda()
        self.identity = identity.unsqueeze(0)
        self.mask = torch.ones((3,3)).cuda()
        self.mask[2,2] = 0
    
    def forward(self, inputs, labels):
        warp = torch.rand((len(labels), 3, 3)).cuda() * self.bound
        warp.requires_grad_()
        #print('input size')
        #print(inputs.size(2), inputs.size(3))
        new_data = kornia.geometry.transform.warp_perspective(inputs, (self.mask * warp) + self.identity, (inputs.size(2), inputs.size(3)))
        #print('data size')
        #print(new_data.size(2), new_data.size(3))
        for i in range(self.num_iterations):
            outputs = self.model(new_data)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            self.model.zero_grad()
            cost = F.cross_entropy(outputs, labels)
            if i + 1 == self.num_iterations:
                warp_grad = torch.autograd.grad(cost, warp, retain_graph=True)[0]
            else:
                warp_grad = torch.autograd.grad(cost, warp, retain_graph=False)[0]
            warp = torch.clamp(warp + torch.sign(warp_grad) * self.step_size, -self.bound,
                              self.bound).detach().requires_grad_()
            new_data = kornia.geometry.transform.warp_perspective(inputs, (self.mask * warp) + self.identity, (inputs.size(2), inputs.size(3)))
        #print(new_data.size(2), new_data.size(3))
        return new_data

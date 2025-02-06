import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import device
from .attacks import AttackInstance, l2_loss


def get_bar_variables(image, num_bars, bar_width, grey):
    batch_size, num_channels, height, width = image.shape

    bar_locations = torch.arange(
        math.floor(width / num_bars) // 2,
        width - math.floor(width / num_bars) // 2,
        width // num_bars,
    )

    xx, yy = torch.meshgrid(
        torch.arange(0, width), torch.arange(0, height), indexing="xy"
    )

    mask = torch.zeros((height, width), dtype=torch.bool).to(device)

    for b in bar_locations:
        mask += torch.logical_and(
            ((b - bar_width // 2) <= xx), (xx <= (b + bar_width // 2))
        ).to(device)

    bar_variables = torch.zeros(
        image.size(), device=device, requires_grad=True
    ).to(device)

    return bar_variables, mask


def add_bars(inputs, mask, grey):
    inputs = inputs.clone().detach()

    inputs[:, :, mask] = grey.view(1, 3, 1)

    return inputs


def apply_bar_variables(inputs, bar_variables, mask):
    return inputs + bar_variables * mask


class PrisonAdversary(nn.Module):
    """Implementation of the Prison attack, which adds "bars" to the image, who's colour values
    can then be changed arbitrarily.

    Parameters
    ---
    epsilon (float):
        epsilon used in attack

    num_steps (int):
        number of steps used in the opitimsation loop

    step_size (flaot):
        step size used in the optimisation loop

    num_bars (int):
        number of bars used in the attack

    bar_width (int):
        width of the bars used in the attack
    """

    def __init__(self, epsilon, num_steps, step_size, num_bars, bar_width, task):
        super().__init__()

        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_bars = num_bars
        self.bar_width = bar_width
        self.grey = torch.tensor([0.5, 0.5, 0.5]).to(device)
        self.task = task

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)
        clean_out = model(inputs)
        """

        The attack works by taking an image an blurring it, and then optimally interpolating pixel-wise between the blurred image
        and the original.

        model: the model to be attacked.
        inputs: batch of unmodified images.
        targets: true labels.

        returns: adversarially perturbed images.
        """

        inputs, targets = inputs.clone().to(device), targets.to(device)

        # We have a mask to control what is optimised, and varialbes which control the colour of the bars
        bar_variables, mask = get_bar_variables(
            inputs, self.num_bars, self.bar_width, self.grey
        )
        if self.task == "var_reg":
            bar_variables_2, mask_2 = get_bar_variables(
                inputs, self.num_bars, self.bar_width, self.grey
            )
        inputs = add_bars(inputs, mask, self.grey)

        # The inner loop
        for i in range(self.num_steps):
            adv_inputs = apply_bar_variables(inputs, bar_variables, mask)

            outputs = model(adv_inputs)
            if self.task == "attack":
                loss = F.cross_entropy(outputs, targets)
            elif self.task == "l2":
                loss = l2_loss(outputs, clean_out)
            else:
                adv_inputs_2 = apply_bar_variables(inputs, bar_variables_2, mask_2)
                loss = l2_loss(outputs, model(adv_inputs_2))

            # Typical PGD implementation
            if self.task == "var_reg":
                grad, grad2 = torch.autograd.grad(loss, [bar_variables, bar_variables_2], only_inputs=True)
            else:
                grad = torch.autograd.grad(loss, bar_variables, only_inputs=True)[0]
            grad = torch.sign(grad)

            bar_variables = bar_variables + self.step_size * grad
            bar_variables = torch.clamp(bar_variables, 0, self.epsilon)

            bar_variables = bar_variables.detach()
            bar_variables.requires_grad = True

            if self.task == 'var_reg':
                bar_variables_2 = bar_variables_2 + self.step_size * grad2
                bar_variables_2 = torch.clamp(bar_variables_2, 0, self.epsilon)

                bar_variables_2 = bar_variables_2.detach()
                bar_variables_2.requires_grad = True

        adv_inputs = apply_bar_variables(inputs, bar_variables, mask)
        if self.task != "attack":
            if self.task == "var_reg":
                adv_inputs_2 = apply_bar_variables(inputs, bar_variables_2, mask_2)
                return l2_loss(model(adv_inputs.detach()), model(adv_inputs_2.detach()))
            else:
                return l2_loss(model(adv_inputs.detach()), clean_out)

        return adv_inputs

class PrisonAttackBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, prison_num_bars, prison_bar_width):
        super().__init__(model, 0)
        self.attack = PrisonAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            num_bars=prison_num_bars,
            bar_width=prison_bar_width,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class PrisonRegBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, prison_num_bars, prison_bar_width, task):
        super().__init__(model, 0)
        self.attack = PrisonAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            num_bars=prison_num_bars,
            bar_width=prison_bar_width,
            task=task
        )

    def get_reg_term(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class PrisonAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = PrisonAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            num_bars=args.prison_num_bars,
            bar_width=args.prison_bar_width,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return PrisonAttack(model, args)

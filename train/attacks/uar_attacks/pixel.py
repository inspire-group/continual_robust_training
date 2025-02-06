import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import AttackInstance, normalize_l2, tensor_clamp_l2, l2_loss
from .config import device


def get_pixels(image, pixel_size):
    """
    Pixelates an image by selecting every pixel_size numebered pixel.
    """

    pixelated_image = image[..., ::pixel_size, ::pixel_size].clone().detach()
    return pixelated_image


def pixelate_image(image, pixelated_image, pixel_variables):
    """
    Pixelates the image, depending on the value of each of the entries in pixel_variables.
    """

    batch_size, num_channels, pixelated_height, pixelated_width = pixelated_image.size()
    _, _, height, width = image.size()
    pixel_size = height // pixelated_height

    # Take the image and put it into blocks, then average the blocks, and return to the original iamge (note the averaging is done
    # by interpolating to the pixel function, which is equivalent)
    chunked_image = (
        image.view(
            batch_size,
            num_channels,
            pixelated_height,
            pixel_size,
            pixelated_width,
            pixel_size,
        )
        .transpose(3, 4)
        .contiguous()
    )
    pixel_variables = pixel_variables.unsqueeze(-1).unsqueeze(-1)
    return_image = (
        chunked_image * (1 - pixel_variables)
        + (pixelated_image.unsqueeze(-1).unsqueeze(-1)) * pixel_variables
    )
    return_image = (
        return_image.transpose(3, 4)
        .contiguous()
        .view(batch_size, num_channels, height, width)
    )

    return return_image.clamp(0, 1)


class PixelAdversary(nn.Module):
    """
    This carries out the Pixel attack, which works by pixelating the image in a differentiable manner.

    Parameters
    ----

    epislon (float):
         epsilon used in attack

    num_steps (int):
         number of steps used in the opitimsation loop

    step_size (flaot):
         step size used in the optimisation loop

    pixel_size (int):
        size of the pixels used in the pixelation

    distance_metric(str):
        distance metric used in the attack, either 'l2' or 'linf'
    """

    def __init__(self, epsilon, num_steps, step_size, distance_metric, pixel_size, task="attack"):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.pixel_size = pixel_size
        self.distance_metric = distance_metric
        self.task = task

        if distance_metric == "l2":
            self.normalise_tensor = lambda x: normalize_l2(x)
            self.project_tensor = lambda x, epsilon: tensor_clamp_l2(
                x, 0, epsilon
            )
        elif distance_metric == "linf":
            self.project_tensor = lambda x, epsilon: torch.clamp(x, -epsilon, epsilon)
            self.normalise_tensor = lambda x: torch.sign(x)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {distance_metric}"
            )

    def forward(self, model, inputs, targets):
        """
        Implements the pixel attack, which works by pixelating the image in a differentiable manner.
        """
        batch_size, num_channels, height, width = inputs.size()
        clean_out = model(inputs)

        even_pixelation = not (
            height % self.pixel_size == 0 and width % self.pixel_size == 0
        )
        if even_pixelation:
            inputs = F.interpolate(
                inputs,
                size=(
                    math.ceil(height / self.pixel_size) * self.pixel_size,
                    math.ceil(width / self.pixel_size) * self.pixel_size,
                ),
                mode="bilinear",
                align_corners=False,
            )

        pixelated_image = get_pixels(inputs, self.pixel_size)

        pixel_vars = (
            self.epsilon
            * random.random()
            * self.normalise_tensor(
                torch.rand(
                    batch_size,
                    num_channels,
                    math.ceil(height / self.pixel_size),
                    math.ceil(width / self.pixel_size),
                    device=device,
                )
            )
        )

        pixel_vars = torch.abs(pixel_vars).clamp(0, 1)
        pixel_vars.requires_grad = True

        if self.task == "var_reg":
            pixel_vars_2 = (
                self.epsilon
                * random.random()
                * self.normalise_tensor(
                    torch.rand(
                        batch_size,
                        num_channels,
                        math.ceil(height / self.pixel_size),
                        math.ceil(width / self.pixel_size),
                        device=device,
                    )
                )
            )

            pixel_vars_2 = torch.abs(pixel_vars_2).clamp(0, 1)
            pixel_vars_2.requires_grad = True

        for _ in range(0, self.num_steps):
            adv_inputs = pixelate_image(inputs, pixelated_image, pixel_vars)

            if even_pixelation:
                adv_inputs = adv_inputs[..., :height, :width]

            logits = model(adv_inputs)

            if self.task == "attack":
                loss = F.cross_entropy(logits, targets)
            elif self.task == "l2":
                loss = l2_loss(logits, clean_out)
            else:
                adv_inputs_2 = pixelate_image(inputs, pixelated_image, pixel_vars_2)
                if even_pixelation:
                    adv_inputs_2 = adv_inputs_2[..., :height, :width]
                loss = l2_loss(logits, model(adv_inputs_2))

            if self.task == "var_reg":
                grad, grad2 = torch.autograd.grad(loss, [pixel_vars, pixel_vars_2], only_inputs=True)
            else:
                grad = torch.autograd.grad(loss, pixel_vars, only_inputs=True)[0]

            pixel_vars = pixel_vars + self.step_size * self.normalise_tensor(grad)
            pixel_vars = self.project_tensor(pixel_vars, self.epsilon)
            pixel_vars = pixel_vars.clamp(0, 1)
            pixel_vars = pixel_vars.detach()
            pixel_vars.requires_grad = True

            if self.task == "var_reg":
                pixel_vars_2 = pixel_vars_2 + self.step_size * self.normalise_tensor(grad2)
                pixel_vars_2 = self.project_tensor(pixel_vars_2, self.epsilon)
                pixel_vars_2 = pixel_vars_2.clamp(0, 1)
                pixel_vars_2 = pixel_vars_2.detach()
                pixel_vars_2.requires_grad = True

        adv_inputs = pixelate_image(inputs, pixelated_image, pixel_vars)

        if even_pixelation:
            adv_inputs = adv_inputs[..., :height, :width]

        if self.task != "attack":
            if self.task == "var_reg":
                adv_inputs_2 = pixelate_image(inputs, pixelated_image, pixel_vars_2)

                if even_pixelation:
                    adv_inputs_2 = adv_inputs_2[..., :height, :width]
                return l2_loss(model(adv_inputs.detach()), model(adv_inputs_2.detach()))
            else:
                return l2_loss(model(adv_inputs.detach()), clean_out)

        return adv_inputs.detach()

class PixelAttackBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, pixel_size, distance_metric):
        super().__init__(model, 0)
        self.attack = PixelAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            pixel_size=pixel_size,
            distance_metric=distance_metric,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class PixelRegBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, pixel_size, distance_metric, task):
        super().__init__(model, 0)
        self.attack = PixelAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            pixel_size=pixel_size,
            distance_metric=distance_metric,
            task=task
        )
        self.model = model

    def get_reg_term(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


class PixelAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = PixelAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            pixel_size=args.pixel_size,
            distance_metric=args.distance_metric,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return PixelAttack(model, args)

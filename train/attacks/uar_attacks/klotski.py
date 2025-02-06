import torch

from .attacks import AttackInstance, normalize_l2, tensor_clamp_l2, l2_loss
import random
from .config import device
import torch.nn.functional as F
import torch.nn as nn


def shift_image(image, shift_vars):
    _, shift_height, shift_width, _ = shift_vars.shape
    batch_size, channels, height, width = image.shape

    num_repeats_height, num_repeats_width = height // shift_height, width // shift_width

    base_coords = torch.stack(
        torch.meshgrid(torch.arange(height), torch.arange(width), indexing="xy"),
        axis=-1,
    )
    base_coords = (
        (base_coords / torch.tensor([height, width], dtype=torch.float32)) - 0.5
    ) * 2
    base_coords = base_coords.to(device)

    shifted_coords_repeat = torch.repeat_interleave(
        shift_vars, num_repeats_height, axis=1
    )
    shifted_coords_repeat = torch.repeat_interleave(
        shifted_coords_repeat, num_repeats_width, axis=2
    )

    shifted_coords = base_coords + shifted_coords_repeat

    return torch.nn.functional.grid_sample(image, shifted_coords, padding_mode="zeros")


class KlotskiAdversary(nn.Module):

    """
    Implements the Klotski attack, which works by cutting the image into blocks and shifting them around.

    Arguments
    ---

    epsilon: float
        The maximum distortion on each block

    num_steps: int
        The number of steps to take in the attack

    step_size: float
        The step size to take in the attack

    distance_metric: str
        The distance metric to use, either 'l2' or 'linf'
    """

    def __init__(self, epsilon, num_steps, step_size, distance_metric, num_blocks, task):
        super(KlotskiAdversary, self).__init__()

        self.step_size = step_size
        self.distance_metric = distance_metric
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.num_blocks = num_blocks
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
        batch_size, num_channels, height, width = inputs.shape
        clean_out = model(inputs)
        if self.distance_metric == "l2":
            shift_vars = (
                self.epsilon
                * self.normalise_tensor(
                    torch.rand(
                        batch_size,
                        self.num_blocks,
                        self.num_blocks,
                        2,
                        device=device,
                    )
                )
                * random.random()
            )
            if self.task == "var_reg":
                shift_vars_2 = (
                    self.epsilon
                    * self.normalise_tensor(
                        torch.rand(
                            batch_size,
                            self.num_blocks,
                            self.num_blocks,
                            2,
                            device=device,
                        )
                    )
                    * random.random()
                )
                shift_vars_2.requires_grad = True
        if self.distance_metric == "linf":
            shift_vars = self.epsilon * torch.rand(
                batch_size, self.num_blocks, self.num_blocks, 2, device=device
            )
            if self.task == "var_reg":
                shift_vars_2 = self.epsilon * torch.rand(
                    batch_size, self.num_blocks, self.num_blocks, 2, device=device
                )
                shift_vars_2.requires_grad = True

        shift_vars.requires_grad = True

        for _ in range(0, self.num_steps):
            adv_inputs = shift_image(inputs, shift_vars)

            logits = model(adv_inputs)
            if self.task == "attack":
                loss = F.cross_entropy(logits, targets)
            elif self.task == "l2":
                loss = l2_loss(logits, clean_out)
            else:
                adv_inputs_2 = shift_image(inputs, shift_vars_2)
                loss = l2_loss(logits, model(adv_inputs_2))

            if self.task == "var_reg":
                grad, grad2 = torch.autograd.grad(loss, [shift_vars, shift_vars_2], only_inputs=True)
            else:
                grad = torch.autograd.grad(loss, shift_vars, only_inputs=True)[0]

            shift_vars = shift_vars + self.step_size * self.normalise_tensor(
                grad.unsqueeze(1)
            ).squeeze(1)
            shift_vars = self.project_tensor(
                shift_vars.unsqueeze(1), self.epsilon
            ).squeeze(1)
            shift_vars = shift_vars.detach()
            shift_vars.requires_grad = True

            if self.task == "var_reg":
                shift_vars_2 = shift_vars_2 + self.step_size * self.normalise_tensor(
                    grad2.unsqueeze(1)
                ).squeeze(1)
                shift_vars_2 = self.project_tensor(
                    shift_vars_2.unsqueeze(1), self.epsilon
                ).squeeze(1)
                shift_vars_2 = shift_vars_2.detach()
                shift_vars_2.requires_grad = True

        adv_inputs = shift_image(inputs, shift_vars)

        if self.task != "attack":
            if self.task == "var_reg":
                adv_inputs_2 = adv_inputs_2 = shift_image(inputs, shift_vars_2)
                return l2_loss(model(adv_inputs.detach()), model(adv_inputs_2.detach()))
            else:
                return l2_loss(model(adv_inputs.detach()), clean_out)

        return adv_inputs.detach()

class KlotskiAttackBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, distance_metric, klotski_num_blocks):
        super().__init__(model, 0)
        self.attack = KlotskiAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            distance_metric=distance_metric,
            num_blocks=klotski_num_blocks,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class KlotskiRegBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, distance_metric, klotski_num_blocks, task):
        super().__init__(model, 0)
        self.attack = KlotskiAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            distance_metric=distance_metric,
            num_blocks=klotski_num_blocks,
            task=task
        )
        self.model = model

    def get_reg_term(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class KlotskiAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = KlotskiAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            num_blocks=args.klotski_num_blocks,
        )
        self.model = model

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return KlotskiAttack(model, args)

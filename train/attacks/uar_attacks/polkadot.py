import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .config import device
from .attacks import AttackInstance, normalize_l2, tensor_clamp_l2, l2_loss


def apply_strengths(
    inputs,
    centers,
    variables,
    colour_images,
    distance_scaling,
    image_threshold,
    distance_normaliser,
):
    """
    Given some variables describing centers, return the "strength" of each center for each input.
    """
    batch_size, channels, height, width = inputs.size()

    xx, yy = torch.meshgrid(
        torch.arange(0, height), torch.arange(0, width), indexing="xy"
    )
    xx, yy = xx / width, yy / height
    xx, yy = xx.to(device), yy.to(device)
    centers = centers.unsqueeze(0).unsqueeze(0)

    distances = (
        1
        - torch.sqrt(
            (xx.unsqueeze(-1) - centers[..., 0]) ** 2
            + (yy.unsqueeze(-1) - centers[..., 1]) ** 2
        )
    ) ** distance_normaliser
    distances_scaled = distances.unsqueeze(0) * variables.unsqueeze(1).unsqueeze(1)
    distances_scaled = (
        torch.concat(
            [
                torch.ones_like(distances_scaled[..., 0:1], device=device)
                * image_threshold,
                distances_scaled,
            ],
            dim=-1,
        )
        * distance_scaling
    )
    distances_softmax = torch.softmax(distances_scaled, dim=-1)

    interpolation_images = torch.concat([inputs.unsqueeze(-1), colour_images], dim=-1)
    return_images = interpolation_images * distances_softmax.unsqueeze(1)

    return torch.sum(return_images, dim=-1)


class PolkadotAdversary(nn.Module):
    """Implemetnation of the polkadot adversary, which works by adding polkadots to an image,
    and then optimisng their size."""

    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        distance_metric,
        num_polkadots,
        distance_scaling,
        image_threshold,
        distance_normaliser,
        task
    ):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.distance_metric = distance_metric
        self.num_polkadots = num_polkadots
        self.image_threshold = image_threshold
        self.distance_scaling = distance_scaling
        self.distance_normaliser = distance_normaliser
        self.task = task

        if distance_metric == "l2":
            self.normalise_tensor = lambda x: normalize_l2(x)
            self.project_tensor = lambda x, epsilon: torch.abs(
                tensor_clamp_l2(x, 0, epsilon)
            )
        elif distance_metric == "linf":
            self.project_tensor = lambda x, epsilon: torch.abs(
                torch.clamp(x, -epsilon, epsilon)
            )
            self.normalise_tensor = lambda x: torch.sign(x)
        else:
            raise ValueError(
                f"Distance metric must be either 'l2' or 'inf',was {distance_metric}"
            )

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, num_channels, height, width = inputs.size()
        clean_out = model(inputs)

        # We initalise the interpolation matrix, on which the inner loop performs PGD.

        strength_vars = torch.rand(
            batch_size, self.num_polkadots, requires_grad=True, device=device
        )

        if self.task == "var_reg":
            strength_vars_2 = torch.rand(
                batch_size, self.num_polkadots, requires_grad=True, device=device
            )
            
        centers = torch.rand(self.num_polkadots, 2, device=device)
        colours = torch.rand(
            (batch_size, num_channels, 1, 1, self.num_polkadots), device=device
        ).repeat(1, 1, height, width, 1)

        # The inner loop
        for _ in range(self.num_steps):
            adv_inputs = apply_strengths(
                inputs,
                centers,
                strength_vars,
                colours,
                self.distance_scaling,
                self.image_threshold,
                self.distance_normaliser,
            )
            logits = model(adv_inputs)

            if self.task == "attack":
                loss = F.cross_entropy(logits, targets)
            elif self.task == "l2":
                loss = l2_loss(logits, clean_out)
            else:
                adv_inputs_2 = apply_strengths(
                    inputs,
                    centers,
                    strength_vars_2,
                    colours,
                    self.distance_scaling,
                    self.image_threshold,
                    self.distance_normaliser,
                )
                loss = l2_loss(logits, model(adv_inputs_2))

            # Typical PGD implementation
            if self.task == "var_reg":
                grad, grad2 = torch.autograd.grad(loss, [strength_vars, strength_vars_2], only_inputs=True)
            else:
                grad = torch.autograd.grad(loss, strength_vars, only_inputs=True)[0]
            grad = self.normalise_tensor(grad)

            strength_vars = strength_vars + self.step_size * grad
            strength_vars = self.project_tensor(strength_vars, self.epsilon)

            strength_vars = strength_vars.detach()
            strength_vars.requires_grad = True

            if self.task == "var_reg":
                strength_vars_2 = strength_vars_2 + self.step_size * grad2
                strength_vars_2 = self.project_tensor(strength_vars_2, self.epsilon)

                strength_vars_2 = strength_vars_2.detach()
                strength_vars_2.requires_grad = True

        adv_inputs = apply_strengths(
            inputs,
            centers,
            strength_vars,
            colours,
            self.distance_scaling,
            self.image_threshold,
            self.distance_normaliser,
        )

        if self.task != "attack":
            if self.task == "var_reg":
                adv_inputs_2 = apply_strengths(
                    inputs,
                    centers,
                    strength_vars_2,
                    colours,
                    self.distance_scaling,
                    self.image_threshold,
                    self.distance_normaliser,
                )
                return l2_loss(model(adv_inputs.detach()), model(adv_inputs_2.detach()))
            else:
                return l2_loss(model(adv_inputs.detach()), clean_out)

        return adv_inputs

class PolkadotAttackBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, distance_metric, 
    polkadot_num_polkadots, polkadot_distance_scaling, polkadot_image_threshold, polkadot_distance_normaliser):
        super().__init__(model, 0)
        self.attack = PolkadotAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            distance_metric=distance_metric,
            num_polkadots=polkadot_num_polkadots,
            distance_scaling=polkadot_distance_scaling,
            image_threshold=polkadot_image_threshold,
            distance_normaliser=polkadot_distance_normaliser,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class PolkadotRegBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, distance_metric, 
    polkadot_num_polkadots, polkadot_distance_scaling, polkadot_image_threshold, polkadot_distance_normaliser, task):
        super().__init__(model, 0)
        self.attack = PolkadotAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            distance_metric=distance_metric,
            num_polkadots=polkadot_num_polkadots,
            distance_scaling=polkadot_distance_scaling,
            image_threshold=polkadot_image_threshold,
            distance_normaliser=polkadot_distance_normaliser,
            task=task
        )

    def get_reg_term(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class PolkadotAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = PolkadotAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            distance_metric=args.distance_metric,
            num_polkadots=args.polkadot_num_polkadots,
            distance_scaling=args.polkadot_distance_scaling,
            image_threshold=args.polkadot_image_threshold,
            distance_normaliser=args.polkadot_distance_normaliser,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return PolkadotAttack(model, args)

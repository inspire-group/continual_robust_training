import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import device
from .attacks import AttackInstance, l2_loss


def glitch_image(images, glitch_vars, num_lines):
    batch_size, num_channels, width, height = images.shape
    line_width = height // num_lines

    displacement_tensor = torch.repeat_interleave(
        glitch_vars.unsqueeze(-1), line_width, dim=2
    )
    # We split the image into num_lines lines, each one being displaced by one of the variables in glitch_vars

    # We find the displacement corrdinates for the image
    xx, yy = torch.meshgrid(
        torch.arange(0, width), torch.arange(0, height), indexing="xy"
    )
    xx, yy = (xx.repeat(batch_size, num_channels, 1, 1) / width) * 2 - 1, (
        yy.repeat(batch_size, num_channels, 1, 1) / height
    ) * 2 - 1
    xx = xx + displacement_tensor
    new_location = torch.stack([xx, yy], dim=-1).to(device)

    # We use grid_sample to find the new pixel values
    image_channels = []
    for j in range(0, num_channels):
        image_channels.append(
            F.grid_sample(
                images[:, j : j + 1],
                new_location[:, j],
                padding_mode="reflection",
                align_corners=False,
            )
        )

    return_images = torch.concat(image_channels, dim=1)

    return return_images


def grey_images(
    images, grey_strength, grey=torch.tensor((0.87, 0.87, 0.87)).to(device)
):
    """
    Greys out the image by adding a constant to each pixel
    """

    return_images = (
        images * (1 - grey_strength) + grey.unsqueeze(-1).unsqueeze(-1) * grey_strength
    )
    return return_images


class GlitchAdversary(nn.Module):
    """
    Implements the Glitch attack, functoning by splitting the image into horizontal bars and then displacing the colour channels in each bar
    by some random amount.

    Parameters
    ---

    epsilon: float
        The maximum amount of displacement to apply to each bar.

    num_steps: int
        The number of steps to perform in the inner loop of the attack.

    step_size: float
        The step size to use in the inner loop of the attack.

    num_lines: int
        The number of horizontal bars to split the image into.

    grey_strength: float
        The amount of greying to apply to the image before applying the attack.
    """

    def __init__(self, epsilon, num_steps, step_size, num_lines, grey_strength, task="attack"):
        super().__init__()
        self.epsilon = epsilon

        self.num_steps = num_steps
        self.step_size = step_size
        self.num_lines = num_lines
        self.grey_strength = grey_strength
        self.task = task

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)
        clean_out = model(inputs)

        batch_size, num_channels, height, width = inputs.size()
        inputs = grey_images(inputs, self.grey_strength)

        # We initalise the interpolation matrix, on which the inner loop performs PGD.

        glitch_vars = self.epsilon * torch.rand(
            batch_size, num_channels, self.num_lines, requires_grad=True
        )

        if self.task == "var_reg":
            glitch_vars_2 = self.epsilon * torch.rand(
                batch_size, num_channels, self.num_lines, requires_grad=True
            )

        if height % self.num_lines != 0:
            raise ValueError("The number of lines must divide the height of the image")

        # The inner loop
        for i in range(self.num_steps):
            adv_inputs = glitch_image(inputs, glitch_vars, self.num_lines)
            outputs = model(adv_inputs)

            if self.task == "attack":
                loss = F.cross_entropy(outputs, targets)
            elif self.task == "l2":
                loss = l2_loss(outputs, clean_out)
            else:
                adv_inputs_2 = glitch_image(inputs, glitch_vars_2, self.num_lines)
                loss = l2_loss(outputs, model(adv_inputs_2))

            # Typical PGD implementation
            if self.task == "var_reg":
                grad, grad2 = torch.autograd.grad(loss, [glitch_vars, glitch_vars_2], only_inputs=True)
            else:
                grad = torch.autograd.grad(loss, glitch_vars, only_inputs=True)[0]
            grad = torch.sign(grad)

            glitch_vars = glitch_vars + self.step_size * grad
            glitch_vars = torch.clamp(glitch_vars, -self.epsilon, self.epsilon)

            glitch_vars = glitch_vars.detach()
            glitch_vars.requires_grad = True

            if self.task == "var_reg":
                grad2 = torch.sign(grad2)

                glitch_vars_2 = glitch_vars_2 + self.step_size * grad2
                glitch_vars_2 = torch.clamp(glitch_vars_2, -self.epsilon, self.epsilon)

                glitch_vars_2 = glitch_vars_2.detach()
                glitch_vars_2.requires_grad = True

        adv_inputs = glitch_image(inputs, glitch_vars, self.num_lines)
        if self.task != "attack":
            if self.task == "var_reg":
                adv_inputs_2 = glitch_image(inputs, glitch_vars_2, self.num_lines)
                return l2_loss(model(adv_inputs.detach()), model(adv_inputs_2.detach()))
            else:
                return l2_loss(model(adv_inputs.detach()), clean_out)

        return adv_inputs


class GlitchAttackBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, glitch_num_lines, glitch_grey_strength):
        super().__init__(model, 0)
        self.attack = GlitchAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            num_lines=glitch_num_lines,
            grey_strength=glitch_grey_strength,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class GlitchRegBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, glitch_num_lines, glitch_grey_strength, task):
        super().__init__(model, 0)
        self.attack = GlitchAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            num_lines=glitch_num_lines,
            grey_strength=glitch_grey_strength,
            task=task
        )

    def get_reg_term(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class GlitchAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = GlitchAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            num_lines=args.glitch_num_lines,
            grey_strength=args.glitch_grey_strength,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return GlitchAttack(model, args)

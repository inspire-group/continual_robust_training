import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .config import device
from .attacks import AttackInstance, l2_loss


def dynamic_interpolation(
    image, perturbed_image, interpolation_matrix, kernel_size=5, sigma=3
):
    """Performs dynamic interpolation, as described above.

    image: the original input image.
    perturbed_image: the perturbed image which is interpolated with to create the example.
    interpolation_matrix: matrix used to do element-wise interpolation bertween two matrices.

    returns the adversarial example formed by interpolation between the original and perturbed image.
    """
    image = image.detach()
    interpolation_matrix = torchvision.transforms.functional.gaussian_blur(
        interpolation_matrix, kernel_size=kernel_size, sigma=sigma
    )

    adv_img = (
        interpolation_matrix * perturbed_image + (1 - interpolation_matrix) * image
    )
    adv_img = torch.clamp(
        adv_img, 0, 1
    )  # Need to make sure our adversarial image does not violate the pixel range constraints.

    return adv_img


def blur_image(image, kernel_size, sigma):
    """Applies gaussian blur to an image"""

    return torchvision.transforms.functional.gaussian_blur(
        image, kernel_size=kernel_size, sigma=sigma
    )


class BlurAdversary(nn.Module):
    """Implementation of the blur attack, which involves interpolating between an image and its blurred version,
    optimising for the correct level of interpolation.

    Parameters
    ---
    Most parameters are the same when compared to other attacks apart from.

    blur_kernel_size: int
     Size of the gaussian kernel used to add the blur

    sigma: float
      Standard deviation of the gaussian kernel within the blurring

    blur_init: float
       The initialisation value of the blur tensor"""

    def __init__(
        self,
        epsilon,
        num_steps,
        step_size,
        blur_kernel_size,
        blur_sigma,
        interp_kernel_size,
        interp_kernel_sigma,
        task="attack"
    ):
        super().__init__()
        self.epsilon = epsilon

        self.num_steps = num_steps
        self.step_size = step_size

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma

        self.interp_kernel_size = interp_kernel_size
        self.interp_kernel_sigma = interp_kernel_sigma
        self.task = task

    def forward(self, model, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)
        """

        The attack works by taking an image an blurring it, and then optimally interpolating pixel-wise between the blurred image
        and the original.

        model: the model to be attacked.
        inputs: batch of unmodified images.
        targets: true labels.

        returns: adversarially perturbed images.
        """

        batch_size, _, _, width = inputs.size()
        clean_out = model(inputs)

        # We initalise the interpolation matrix, on which the inner loop performs PGD.

        interp_matrix = (
            self.epsilon
            * torch.rand((batch_size, 1, width, width), device=device)
        ).detach()
        perturbed_inputs = blur_image(
            inputs, self.blur_kernel_size, self.blur_sigma
        ).detach()
        interp_matrix.requires_grad = True

        if self.task == "var_reg":
            interp_matrix_2 = (
                self.epsilon
                * torch.rand((batch_size, 1, width, width), device=device)
                ).detach()


        # The inner loop
        for i in range(self.num_steps):
            adv_inputs = dynamic_interpolation(
                inputs,
                perturbed_inputs,
                interp_matrix,
                self.interp_kernel_size,
                self.interp_kernel_sigma,
            )
            if self.task == "var_reg":
                adv_inputs_2 = dynamic_interpolation(
                    inputs,
                    perturbed_inputs,
                    interp_matrix_2,
                    self.interp_kernel_size,
                    self.interp_kernel_sigma,
                    )
            logits = model(adv_inputs)
            if self.task == "attack":
                loss = F.cross_entropy(logits, targets)
            elif self.task == "l2":
                loss = l2_loss(logits, clean_out)
            else:
                loss = l2_loss(logits, model(adv_inputs_2))


            # Typical PGD implementation
            if self.task == "var_reg":
                grad, grad_2 = torch.autograd.grad(loss, [interp_matrix, interp_matrix_2], only_inputs=True)
            else:
                grad = torch.autograd.grad(loss, interp_matrix, only_inputs=True)[0]
            grad = torch.sign(grad)

            interp_matrix = interp_matrix + self.step_size * grad
            interp_matrix = torch.clamp(interp_matrix, 0, self.epsilon)

            interp_matrix = interp_matrix.detach()
            interp_matrix.requires_grad = True

            if self.task == "var_reg":
                grad_2 = torch.sign(grad_2)

                interp_matrix_2 = interp_matrix_2 + self.step_size * grad_2
                interp_matrix_2 = torch.clamp(interp_matrix_2, 0, self.epsilon)

                interp_matrix_2 = interp_matrix_2.detach()
                interp_matrix_2.requires_grad = True

        adv_inputs = dynamic_interpolation(
            inputs,
            perturbed_inputs,
            interp_matrix,
            self.interp_kernel_size,
            self.interp_kernel_sigma,
        )
        if self.task != "attack":
            if self.task == "var_reg":
                adv_inputs_2 = dynamic_interpolation(
                    inputs,
                    perturbed_inputs,
                    interp_matrix_2,
                    self.interp_kernel_size,
                    self.interp_kernel_sigma,
                    )
                return l2_loss(model(adv_inputs.detach()), model(adv_inputs_2.detach()))
            else:
                return l2_loss(model(adv_inputs.detach()), clean_out)

        return adv_inputs

class BlurAttackBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, blur_kernel_size,
    blur_sigma, interp_kernel_size, interp_kernel_sigma):
        super().__init__(model, 0)
        self.attack = BlurAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            interp_kernel_size=interp_kernel_size,
            interp_kernel_sigma=interp_kernel_sigma,
            task="attack"
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

class BlurRegBase(AttackInstance):
    def __init__(self, model, epsilon, num_steps, step_size, blur_kernel_size,
    blur_sigma, interp_kernel_size, interp_kernel_sigma, task):
        super().__init__(model, 0)
        self.attack = BlurAdversary(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            interp_kernel_size=interp_kernel_size,
            interp_kernel_sigma=interp_kernel_sigma,
            task=task
        )

    def get_reg_term(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)

def get_attack(model, args):
    return BlurAttack(model, args)


class BlurAttack(AttackInstance):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attack = BlurAdversary(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            step_size=args.step_size,
            blur_kernel_size=args.blur_kernel_size,
            blur_sigma=args.blur_kernel_sigma,
            interp_kernel_size=args.blur_interp_kernel_size,
            interp_kernel_sigma=args.blur_interp_kernel_sigma,
        )

    def generate_attack(self, batch):
        xs, ys = batch
        return self.attack(self.model, xs, ys)


def get_attack(model, args):
    return BlurAttack(model, args)

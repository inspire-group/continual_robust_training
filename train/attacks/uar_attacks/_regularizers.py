from .blur import BlurRegBase
from .edge import EdgeRegBase
from .elastic import ElasticRegBase
from .fog import FogRegBase
from .gabor import GaborRegBase
from .glitch import GlitchRegBase
from .hsv import HsvRegBase
from .jpeg import JPEGRegBase
from .kaleidoscope import KaleidoscopeRegBase
from .klotski import KlotskiRegBase
from .mix import MixRegBase
from .pixel import PixelRegBase
from .polkadot import PolkadotRegBase
from .prison import PrisonRegBase
from .snow import SnowRegBase
from .texture import TextureRegBase
from .whirlpool import WhirlpoolRegBase
from .wood import WoodRegBase
import torch.nn as nn
import torch

class BlurReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound=0.3, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            step_size = 0.015
            num_iterations = 20
        else: # imagenet
            step_size = 0.03
            num_iterations = 40
        blur_kernel_size = 17
        blur_kernel_sigma = 25
        blur_interp_kernel_size = 9
        blur_interp_kernel_sigma = 18
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "blur_kernel_size" in kwargs:
            blur_kernel_size = kwargs["blur_kernel_size"]
        if "blur_kernel_sigma" in kwargs:
            blur_kernel_sigma = kwargs["blur_kernel_sigma"]
        if "blur_interp_kernel_size" in kwargs:
            blur_interp_kernel_size = kwargs["blur_interp_kernel_size"]
        if "blur_interp_kernel_sigma" in kwargs:
            blur_interp_kernel_sigma = kwargs["blur_interp_kernel_sigma"]
        
        self.base_attack = BlurRegBase(model, bound, num_iterations, step_size, blur_kernel_size,
    blur_kernel_sigma, blur_interp_kernel_size, blur_interp_kernel_sigma, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class EdgeReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound=0.1, **kwargs):
        super().__init__()
        step_size = 0.02
        num_iterations = 60
        edge_threshold = 0.1
        
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "edge_threshold" in kwargs:
            edge_threshold = kwargs["edge_threshold"]
        self.base_attack = EdgeRegBase(model, bound, num_iterations, step_size, edge_threshold, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class ElasticReg(nn.Module):
    def __init__(self, model, task, dataset_name = "imagenet", bound=0.25, **kwargs):
        super().__init__()
        if dataset_name == "imagenet":
            elastic_upsample_factor = 32
            step_size = 0.003
            num_iterations = 100
        else:
            elastic_upsample_factor = 2
            step_size = 0.006
            num_iterations = 30
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "edge_threshold" in kwargs:
            elastic_upsample_factor = kwargs["elastic_upsample_factor"]
        self.base_attack = ElasticRegBase(model, bound, num_iterations, step_size, elastic_upsample_factor, task)
    
    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class FBMReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.04, **kwargs):
        super().__init__()
        step_size = 0.06
        num_iterations = 30

        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]

        self.base_attack = FbmRegBase(model, bound, num_iterations, step_size, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class FogReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.4, **kwargs):
        super().__init__()
        step_size = 0.05
        if dataset_name == "cifar":
            num_iterations = 40
        else:
            num_iterations = 80
        fog_wibbledecay = 3
        distance_metric = "linf"
        
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "fog_wibbledecay" in kwargs:
            fog_wibbledecay = kwargs["fog_wibbledecay"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        
        self.base_attack = FogRegBase(model, bound, num_iterations, step_size, distance_metric, fog_wibbledecay, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class GaborReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.03, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            gabor_sides = 3
            num_iterations = 80
        else:
            gabor_sides = 5
            num_iterations = 100
        
        step_size = 0.0025
        gabor_kernel_size = 23
        gabor_sigma = 3
        gabor_weight_density = 0.1
        distance_metric = "linf"

        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        if "gabor_sides" in kwargs:
            gabor_sides = kwargs["gabor_sides"]
        if "gabor_kernel_size" in kwargs:
            gabor_kernel_size = kwargs["gabor_kernel_size"]
        if "gabor_sigma" in kwargs:
            gabor_sigma = kwargs["gabor_sigma"]
        if "gabor_weight_density" in kwargs:
            gabor_weight_density = kwargs["gabor_weight_density"]
        
        print("num_iterations", num_iterations)
        self.base_attack = GaborRegBase(model, bound, num_iterations, step_size, distance_metric,
    gabor_kernel_size, gabor_sides, gabor_sigma, gabor_weight_density, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class GlitchReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.05, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            glitch_num_lines = 16
            step_size = 0.0025
            num_iterations = 60
        else:
            glitch_num_lines = 56
            step_size = 0.005
            num_iterations = 90

        glitch_grey_strength = 0

        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "glitch_num_lines" in kwargs:
            glitch_num_lines = kwargs["glitch_num_lines"]
        if "glitch_grey_strength" in kwargs:
            glitch_grey_strength = kwargs["glitch_grey_strength"]

        self.base_attack = GlitchRegBase(model, bound, num_iterations, step_size, glitch_num_lines, glitch_grey_strength, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class HSVReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.02, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            step_size = 0.003
            num_iterations = 20
        else:
            step_size = 0.012
            num_iterations = 50
        hsv_sigma = 3
        hsv_kernel_size = 3
        distance_metric = 'linf'

        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "hsv_sigma" in kwargs:
            hsv_sigma = kwargs["hsv_sigma"]
        if "hsv_kernel_size" in kwargs:
            hsv_kernel_size = kwargs["hsv_kernel_size"]
        
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        

        self.base_attack = HSVRegBase(model, bound, num_iterations, step_size, distance_metric, 
        hsv_kernel_size, hsv_sigma, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class JPEGReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 3/255, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            num_iterations = 50
        else:
            num_iterations = 80
        step_size = 0.0024
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]

        self.base_attack = JPEGRegBase(model, bound, num_iterations, step_size, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class KaleidoscopeReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.1, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            kaleidoscope_num_shapes = 5
            kaleidoscope_shape_size = 4
            kaleidoscope_transparency = 0.2
            kaleidoscope_min_value_valence = 0.8
            kaleidoscope_min_value_saturation = 0.7
            kaleidoscope_edge_width = 5
            num_iterations = 30
        else:
            kaleidoscope_num_shapes = 40
            kaleidoscope_shape_size = 15
            kaleidoscope_transparency = 0.1
            kaleidoscope_min_value_valence = 0.8
            kaleidoscope_min_value_saturation = 0.7
            kaleidoscope_edge_width = 5
            num_iterations = 90
        step_size = 0.005

        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "kaleidoscope_num_shapes" in kwargs:
            kaleidoscope_num_shapes = kwargs["kaleidoscope_num_shapes"]
        if "kaleidoscope_shape_size" in kwargs:
            kaleidoscope_shape_size = kwargs["kaleidoscope_shape_size"]
        if "kaleidoscope_transparency" in kwargs:
            kaleidoscope_transparency = kwargs["kaleidoscope_transparency"]
        if "kaleidoscope_min_value_valence" in kwargs:
            kaleidoscope_min_value_valence = kwargs["kaleidoscope_min_value_valence"]
        if "kaleidoscope_min_value_saturation" in kwargs:
            kaleidoscope_min_value_saturation = kwargs["kaleidoscope_min_value_saturation"]
        if "kaleidoscope_edge_width" in kwargs:
            kaleidoscope_edge_width = kwargs["kaleidoscope_edge_width"]

        self.base_attack = KaleidoscopeRegBase(model, bound, num_iterations, step_size, kaleidoscope_num_shapes, kaleidoscope_shape_size,
    kaleidoscope_min_value_valence, kaleidoscope_min_value_saturation, kaleidoscope_transparency, kaleidoscope_edge_width, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class KlotskiReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.05, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            step_size = 0.005
        else:
            step_size = 0.01
        num_iterations = 50
        klotski_num_blocks = 8
        distance_metric = 'linf'
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "klotski_num_blocks" in kwargs:
            klotski_num_blocks = kwargs["klotski_num_blocks"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        self.base_attack = KlotskiRegBase(model, bound, num_iterations, step_size, distance_metric, klotski_num_blocks, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class MixReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 5, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            step_size = 0.5
            num_iterations = 30
            
        else: 
            step_size = 1.0
            num_iterations = 70
            
        mix_interp_kernel_size = 11
        mix_interp_kernel_sigma = 5
        distance_metric = "l2"

        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "mix_interp_kernel_size" in kwargs:
            mix_interp_kernel_size = kwargs["mix_interp_kernel_size"]
        if "mix_interp_kernel_sigma" in kwargs:
            mix_interp_kernel_sigma = kwargs["mix_interp_kernel_sigma"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        self.base_attack = MixRegBase(model, bound, num_iterations, step_size, distance_metric, mix_interp_kernel_size, mix_interp_kernel_sigma, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class PixelReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 5, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            pixel_size = 4
            num_iterations = 60
        else:
            pixel_size = 16
            num_iterations = 100
        step_size = 1.0
        distance_metric = "l2"
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "pixel_size" in kwargs:
            pixel_size = kwargs["pixel_size"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        self.base_attack = PixelRegBase(model, bound, num_iterations, step_size, pixel_size, distance_metric, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class PolkadotReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 2, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            num_iterations = 40
        else:
            num_iterations = 70
        step_size = 0.3
        polkadot_num_polkadots = 20
        polkadot_distance_scaling = 5
        polkadot_image_threshold = 0.5
        polkadot_distance_normaliser = 5
        distance_metric = "l2"
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "polkadot_num_polkadots" in kwargs:
            polkadot_num_polkadots = kwargs["polkadot_num_polkadots"]
        if "polkadot_distance_scaling" in kwargs:
            polkadot_distance_scaling = kwargs["polkadot_distance_scaling"]
        if "polkadot_image_threshold" in kwargs:
            polkadot_image_threshold = kwargs["polkadot_image_threshold"]
        if "polkadot_distance_normaliser" in kwargs:
            polkadot_distance_normaliser = kwargs["polkadot_distance_normaliser"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        self.base_attack = PolkadotRegBase(model, bound, num_iterations, step_size, distance_metric, 
    polkadot_num_polkadots, polkadot_distance_scaling, polkadot_image_threshold, polkadot_distance_normaliser, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class PrisonReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.03, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            prison_num_bars = 8
            prison_bar_width = 1
            num_iterations = 30
        else:
            prison_num_bars = 7
            prison_bar_width = 6
            num_iterations = 20
        step_size = 0.0015
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "prison_num_bars" in kwargs:
            prison_num_bars = kwargs["prison_num_bars"]
        if "prison_bar_width" in kwargs:
            prison_bar_width = kwargs["prison_bar_width"]
        self.base_attack = PrisonRegBase(model, bound, num_iterations, step_size, prison_num_bars, prison_bar_width, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class SnowReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 4, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            snow_flake_size = 3
            snow_num_layers = 5
            snow_grid_size = 10
            snow_init = 5
            snow_image_discolour = 0.15
            snow_normalizing_constant = 4
            snow_kernel_size = 1
            snow_sigma_range_lower = 10
            snow_sigma_range_upper = 15
            step_size = 0.2
            num_iterations = 20
            
        else:
            snow_flake_size = 15
            snow_num_layers = 5
            snow_grid_size = 15
            snow_init = 5
            snow_image_discolour = 0.15
            snow_normalizing_constant = 4
            snow_kernel_size = 3
            snow_sigma_range_lower = 10
            snow_sigma_range_upper = 15
            step_size = 0.1
            num_iterations = 100
        distance_metric = "l2"
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]

        if "snow_flake_size" in kwargs:
            snow_flake_size = kwargs["snow_flake_size"]
        if "snow_num_layers" in kwargs:
            snow_num_layers = kwargs["snow_num_layers"]
        if "snow_grid_size" in kwargs:
            snow_grid_size = kwargs["snow_grid_size"]
        if "snow_init" in kwargs:
            snow_init = kwargs["snow_init"]
        if "snow_image_discolour" in kwargs:
            snow_image_discolour = kwargs["snow_image_discolour"]
        if "snow_normalizing_constant" in kwargs:
            snow_normalizing_constant = kwargs["snow_normalizing_constant"]
        if "snow_kernel_size" in kwargs:
            snow_kernel_size = kwargs["snow_kernel_size"]
        if "snow_sigma_range_lower" in kwargs:
            snow_sigma_range_lower = kwargs["snow_sigma_range_lower"]
        if "snow_sigma_range_upper" in kwargs:
            snow_sigma_range_upper = kwargs["snow_sigma_range_upper"]

        self.base_attack = SnowRegBase(model, bound, distance_metric, num_iterations, step_size, snow_flake_size, snow_num_layers,
    snow_grid_size, snow_init, snow_image_discolour, snow_normalizing_constant, snow_kernel_size,
    snow_sigma_range_lower, snow_sigma_range_upper, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class TextureReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.1, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            step_size = 0.003
            num_iterations = 30
        else:
            step_size = 0.00075
            num_iterations = 80
        texture_threshold = 0.1
        distance_metric = 'linf'

        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        if "texture_threshold" in kwargs:
            texture_threshold = kwargs["texture_threshold"]

        self.base_attack = TextureRegBase(model, bound, num_iterations, step_size, distance_metric, texture_threshold, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class WhirlpoolReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 100, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            step_size = 16
            num_iterations = 50
        else:
            step_size = 4
            num_iterations = 40
        num_whirlpools = 8
        whirlpool_radius = 0.5
        whirlpool_min_strength = 0.5
        distance_metric = "l2"
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        if "num_whirlpools" in kwargs:
            num_whirlpools = kwargs["num_whirlpools"]
        if "whirlpool_radius" in kwargs:
            whirlpool_radius = kwargs["whirlpool_radius"]
        if "whirlpool_min_strength" in kwargs:
            whirlpool_min_strength = kwargs["whirlpool_min_strength"]
        
        self.base_attack = WhirlpoolRegBase(model, bound, num_iterations, step_size, distance_metric, num_whirlpools,
     whirlpool_radius, whirlpool_min_strength, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))

class WoodReg(nn.Module):
    def __init__(self, model, task, dataset_name = "cifar", bound = 0.05, **kwargs):
        super().__init__()
        if dataset_name == "cifar":
            wood_noise_resolution = 16
            wood_num_rings = 5
            step_size = 0.000625
            num_iterations = 70
        else:
            wood_noise_resolution = 32
            wood_num_rings = 4
            step_size = 0.005
            num_iterations = 80
        wood_random_init = False
        wood_normalising_constant = 8
        if "step_size" in kwargs:
            step_size = kwargs["step_size"]
        if "num_iterations" in kwargs:
            num_iterations = kwargs["num_iterations"]
        if "distance_metric" in kwargs:
            distance_metric = kwargs["distance_metric"]
        if "wood_noise_resolution" in kwargs:
            wood_noise_resolution = kwargs["wood_noise_resolution"]
        if "wood_num_rings" in kwargs:
            wood_num_rings = kwargs["wood_num_rings"]
        if "wood_random_init" in kwargs:
            wood_random_init = kwargs["wood_random_init"]
        if "wood_normalising_constant" in kwargs:
            wood_normalising_constant = kwargs["wood_normalising_constant"]
        
        self.base_attack = WoodRegBase(model, bound, num_iterations, step_size, wood_noise_resolution, wood_num_rings, wood_random_init, wood_normalising_constant, task)

    def forward(self, x):
        y = torch.zeros(len(x)) # dummy variable that is not used for computing regularization
        return self.base_attack.get_reg_term((x, y))


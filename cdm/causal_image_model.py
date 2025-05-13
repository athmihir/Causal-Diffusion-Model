from torch import nn
from cdm.scm import EdgeType, Vertex, Edge, SCM
from cdm.constants import IMAGE_CHANNELS, IMAGE_RESOLUTION
from diffusers import DDPMScheduler, UNet2DConditionModel
import torch

class CausalImageModel(nn.Module):
    
    def __init__(self, scm:SCM):
        super().__init__()
        # Our SCM
        self.scm = scm
        # Our Diffusion Based Image Generater
        self.unet = UNet2DConditionModel(
            sample_size=IMAGE_RESOLUTION,  # the target image resolution
            in_channels=IMAGE_CHANNELS,  # the number of input channels, 3 for RGB images
            out_channels=IMAGE_CHANNELS,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            cross_attention_dim=14,
            block_out_channels=(128, 256, 512, 512),  # Roughly matching our basic unet example
            down_block_types=(
                "AttnDownBlock2D",  # a regular ResNet downsampling block
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "AttnUpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a regular ResNet upsampling block
            ),
        )


    def forward(self, I, noisy_x, timestep, vertex_order, reset_scm=True, intervention_dict:dict[Vertex, torch.tensor]=None):
        if reset_scm:
            # Ensure no values are left cached in our SCM.
            self.scm.clear_intermediate_values()
            # Generate the factors from the SCM.
            value_map, value_map_2 = self.scm(I, intervention_dict)
        elif intervention_dict is not None:
            # When we want to perform intervention but keep same U.
            self.scm.clear_endogenous_values()
            value_map, value_map_2 = self.scm(I, intervention_dict)
        else:
            value_map, value_map_2 = self.scm.value_map, self.scm.value_map_2
        # We concatenate the values in the given vertex order.
        generative_factors = [value_map[v] for v in vertex_order]
        ae_factors = [value_map_2[v] for v in vertex_order]
        conditional_labels = torch.cat(generative_factors, dim=-1).unsqueeze(-2)
        # Return the Unet prediction and the generative factors.
        return self.unet(noisy_x, timestep, conditional_labels).sample, generative_factors, ae_factors
        
        
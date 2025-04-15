from diffusers import AutoencoderKL
from torchvision import datasets
import pandas as pd
import torch
from PIL import Image

class AnnotationReader:
    '''Reads the Labels and maintains them in memory'''
    def __init__(self, filename):
        data = []
        indices = []
        with open(filename, 'r') as file:
            n_lines = int(file.readline())
            self.features = file.readline().split()
            for i in range(n_lines):
                line = file.readline().split()
                indices.append(line[0])
                data.append(line[1:])
        self.dataframe = pd.DataFrame(data, columns=self.features, index=indices)
    
    def __getitem__(self, idx):
        return self.dataframe.index[idx], self.dataframe.iloc[idx].to_dict()
    
    def __len__(self):
        return len(self.dataframe)
    
class CustomImageDataset(datasets.VisionDataset):
    '''Dataset that returns the image tensor and labels. Loads them from disk.'''
    def __init__(self, image_dir, annotation_file, transform=None):
        self.annotations = AnnotationReader(annotation_file)
        self.image_dir = image_dir
        self.transform = transform
        super().__init__(image_dir, transform=transform)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        img_file, labels = self.annotations.__getitem__(idx)
        img_tensor = self.transform(Image.open(os.path.join(self.image_dir, img_file)).convert("RGB"))
        return img_tensor, labels
    
class AutoencoderWrapper:
    '''Inference wrapper for Autoencoder used in encoding and decoding image to Latent space'''

    def __init__(self, DEVICE):
        self.vae = AutoencoderKL.from_pretrained("sd-vae-ft-mse/", torch_dtype=torch.float32)
        self.vae.to(DEVICE)
        self.vae.eval()
        self.lsf = 0.18215 # Used by original authors of Stable Diffusion
    
    def encode(self, img_batch):
        with torch.no_grad():
            return self.vae.encode(img_batch).latent_dist.sample() * self.lsf

    def decode(self, latent_img_batch):
        with torch.no_grad():
            return self.vae.decode(latent_img_batch / self.lsf).sample
from torchvision.transforms import ToPILImage
import torch

def visualize_image(image_tensor):
    '''Allows us to visualize an image tensor'''
    to_pil = ToPILImage()
    image_pil = to_pil(image_tensor)
    return image_pil


def mnist_y_labels(y:torch.Tensor):
    '''Takes the raw y tensor and formats it to labels for the MNIST dataset'''
    digit = torch.argmax(y[:, :10], dim=-1)
    bar = torch.argmax(y[:, 10:12], dim=-1)
    color = torch.argmax(y[:, 12:14], dim=-1)
    return [digit, bar, color]
from torchvision.transforms import ToPILImage

def visualize_image(image_tensor):
    '''Allows us to visualize an image tensor'''
    to_pil = ToPILImage()
    image_pil = to_pil(image_tensor)
    return image_pil
from PIL import Image
import torch
import numpy as np

def preprocess_image(image_path, config):
    """
    Preprocess image for the model
    
    Args:
        image_path (str): Path to the input image
        config (dict): Configuration dictionary
        
    Returns:
        PIL.Image: Preprocessed image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize image
    target_size = config['processing']['image_size']
    image = image.resize((target_size, target_size))
    
    return image 
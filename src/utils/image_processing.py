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
        torch.Tensor: Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize image
    target_size = config['processing']['image_size']
    image = image.resize((target_size, target_size))
    
    # Convert to tensor and normalize
    # This is a basic implementation - you might need to adjust based on your specific needs
    image = torch.from_numpy(np.array(image)).float()
    image = image.permute(2, 0, 1)
    image = image / 255.0
    
    return image 
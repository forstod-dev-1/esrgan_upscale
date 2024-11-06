# esrgan_upscale.py

import torch
import cv2
import numpy as np

def load_esrgan_model(model_path="esrgan.pth"):
    """Load the ESRGAN model from the given path."""
    model = torch.load(model_path)
    model.eval()
    return model

def upscale_with_esrgan(input_image_path, output_image_path, model):
    """Use the ESRGAN model to upscale an image."""
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Input image {input_image_path} not found.")

    with torch.no_grad():
        input_tensor = torch.from_numpy(image).unsqueeze(0).float()
        output_tensor = model(input_tensor)
        output_image = output_tensor.squeeze().cpu().numpy()
    
    upscaled_image = np.uint8(output_image)
    cv2.imwrite(output_image_path, upscaled_image)
    print(f"Image upscaled and saved to {output_image_path}")

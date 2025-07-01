import numpy as np
from PIL import Image
import torch

from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100, 100)), # taille fixe
    transforms.ToTensor(),         # tensor float32 entre 0 et 1
])

def preprocess_image(image_np):
    image_pil = Image.fromarray(image_np).convert("RGB")  # on convertit pour que PIL le traite bien
    image_tensor = transform(image_pil).unsqueeze(0)       # (1, 1, 100, 100)
    return image_tensor


def compute_embedding(model, image_tensor):
    with torch.no_grad():
        emb = model.forward_once(image_tensor)
    return emb.squeeze().numpy()

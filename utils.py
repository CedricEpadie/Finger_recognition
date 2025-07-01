from PIL import Image
from rembg import remove
from torchvision import transforms
import torch
import cv2
import numpy as np

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((100, 100)), # taille fixe
    transforms.ToTensor(),         # tensor float32 entre 0 et 1
])

def enhance_contrast_histogram_equalization(image, keep_rgb=True):
    """
    Améliore le contraste avec égalisation d'histogramme
    Garde les 3 canaux RGB pour VGG16 si keep_rgb=True
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)

        if keep_rgb and len(img_array.shape) == 3:
            # Appliquer CLAHE sur chaque canal RGB séparément
            img_enhanced = np.zeros_like(img_array)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

            for i in range(3):  # Pour chaque canal RGB
                img_enhanced[:, :, i] = clahe.apply(img_array[:, :, i])

            return Image.fromarray(img_enhanced)
        else:
            # Version originale pour niveaux de gris
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_gray)
            return Image.fromarray(img_enhanced)
    else:
        if keep_rgb and len(image.shape) == 3:
            img_enhanced = np.zeros_like(image)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

            for i in range(3):
                img_enhanced[:, :, i] = clahe.apply(image[:, :, i])

            return img_enhanced
        else:
            if len(image.shape) == 3:
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img_enhanced = clahe.apply(img_gray)
            return img_enhanced

def remove_background_rembg(image):
    """
    Supprime le fond de l'image avec rembg
    """
    if not isinstance(image, Image.Image):
        if len(image.shape) == 2:
            image = Image.fromarray(image).convert('RGB')
        else:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image_no_bg = remove(image)

    # Convertir RGBA vers RGB avec fond blanc pour VGG16
    if image_no_bg.mode == 'RGBA':
        background = Image.new('RGB', image_no_bg.size, (255, 255, 255))  # Fond blanc
        background.paste(image_no_bg, mask=image_no_bg.split()[-1])  # Utiliser canal alpha comme masque
        return background

    return image_no_bg

def preprocess_image(image_np):
    image_pil = Image.fromarray(image_np).convert("RGB")    # on convertit pour que PIL le traite bien
    image_pil = enhance_contrast_histogram_equalization(image_pil)  # amélioration de contraste
    image_pil = remove_background_rembg(image_pil)  # on supprime le fond
    image_tensor = transform(image_pil).unsqueeze(0)    # (1, 1, 100, 100)
    return image_tensor


def compute_embedding(model, image_tensor):
    with torch.no_grad():
        emb = model.forward_once(image_tensor)
    return emb.squeeze().numpy()

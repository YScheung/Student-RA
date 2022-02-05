import torch
from PIL import Image
import numpy as np
from model import UNET
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image



def image_augmentation(img_path):
    image = np.array(Image.open(img_path).convert("RGB"))
  
    transform = A.Compose([
        A.Resize(height=340, width=512),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),ToTensorV2(),
    ])   

    augmentations = transform(image=image)
    image = augmentations['image'].unsqueeze(0)
    print(image.shape)   

    # Pytorch expects images to be passed in patches when doing predictions 
    # Without unsqueeze, data size is [3,340,512]. After unsqueeze(0), new dimension is added at index 0
    # Size of image becomes [1,3,340,512] and now it could be passed into the model
        
    return image




if __name__ == '__main__':

    in_files = "data/predict_images/1.jpg"  # General performance is accurate (Precise details, such as mask @ the wheel is less accurate)
    out_files = "output_1.jpg"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    checkpoint = torch.load('my_checkpoint.pth.tar',map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])

    image = image_augmentation(in_files)

    output = model(image)


    img1 = output[0]
    print(img1.shape)
    save_image(img1, "output_1.jpg")




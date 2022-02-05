import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch._C import device
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 25
NUM_WORKERS = 2
IMAGE_HEIGHT = 340  # 1280 originally  (Downsized due to GPU memeory constraint)
IMAGE_WIDTH = 512  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)


        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)


        # Typical processes when training on pytorch models
        optimizer.zero_grad()    # To reset the gradient, as we need to re-draw the tangent line in every epoach
        scaler.scale(loss).backward() # To send / back-propagate the error to the parameters (After appling gradient scaling)
        
        scaler.step(optimizer)  

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.

        scaler.update()  # Update the scale for the next iteration 

        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(          # Image augmentation 
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0), 
            A.HorizontalFlip(p=0.5),  # Training image rotation 
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    '''

    Why do we normalize image beforehand ?
    
    To make the model converge faster. Learning agent can get stable gradient feedbacks avoiding scale differences of different dimensions

    When the data is not normalized, the shared weights of the network have different calibrations for different features, which can make the cost function to converge very slowly and ineffectively.
    
    '''


    val_transforms = A.Compose(    # Image augmentation 
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)  # CHANGE OUTPUT CHANEL TO NUMBER OF CLASSES FOR MULTI CLASS MASKS
    loss_fn = nn.BCEWithLogitsLoss()  # cross entropy loss (For multiclass)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )


    scaler = torch.cuda.amp.GradScaler()  # Gradient scaling with pytorch gradient scaler 

    '''

    Why gradient scaling ?

    Scaling the features makes the flow of gradient descent smooth 
    
    and 
    
    helps algorithms quickly reach the minima of the cost function. Without scaling features
    
    '''



    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader,model,optimizer,loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint) # Save checkpoint for every epoach, final checkpoint -> Final epoach train

        # Check accuracy 
        check_accuracy(val_loader, model,device=DEVICE)
        # print some examples to a foler 

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )





if __name__ == "__main__":
    main()  


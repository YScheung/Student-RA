from model import PSPNet
import torch
from data import DataTransform
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt


# load trained model
net = PSPNet(n_classes=21) 
state_dict = torch.load('./pspnet50_9.pth', map_location='cpu')
net.load_state_dict(state_dict) 
net.eval()


color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std) # Transform images before feeding it into the network



image = Image.open("2012_000050.jpg").convert("RGB")
anno_class_image = Image.open("2012_000050.jpg").convert("RGB")

img, anno_class_img = transform('val', image, anno_class_image) 


x = img.unsqueeze(0)
outputs = net(x)
seg_map = outputs[0]


labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

VOC2012_map = np.array([  # Color responding to different classes according to VOC2012 dataset
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128,192,0],
    [0,64,128]
]) 


pred = torch.argmax(seg_map, dim=1)  # Returns the indices of the maximum value of all elements in the input tensor

pred_imgs = [VOC2012_map[p] for p in pred]  

for pred_img in pred_imgs: 
    plt.imshow(pred_img)
    plt.show()
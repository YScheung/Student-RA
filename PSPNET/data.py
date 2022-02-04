import os.path as ops
import torch.utils.data as data
from PIL import Image
from utils.augmentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor
import matplotlib.pyplot as plt


def make_datapath_list(rootpath): # Create data list 
    original_image_template = ops.join(rootpath, 'JPEGImages', '%s.jpg')
    annotated_image_template = ops.join(rootpath, 'SegmentationClass', '%s.png')

    # Training data, Validation Data
    train_ids = ops.join(rootpath, 'ImageSets/Segmentation/train.txt')
    val_ids = ops.join(rootpath, 'ImageSets/Segmentation/val.txt')

    train_img_list = list()
    train_anno_list = list()


    for line in open(train_ids):
        img_id = line.strip()
        img_path = (original_image_template % img_id)
        anno_path = (annotated_image_template % img_id)

        train_img_list.append(img_path)
        train_anno_list.append(anno_path)
    
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_ids):
        img_id = line.strip()
        img_path = (original_image_template % img_id)
        anno_path = (annotated_image_template % img_id)

        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


# Data transform before training and validation 
class DataTransform():
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            "train": Compose([
                Scale(scale=[0.5,1.5]),
                RandomRotation(angle=[-10,10]), # Image augmentation to have more training data  
                RandomMirror(), 
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
                ]),
            "val": Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img) 



class MyDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)  
        return img, anno_class_img 

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = Image.open(img_file_path)

        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)

        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img
        


if __name__ == "__main__":
    rootpath = "./data/VOC2012"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    color_mean = (0.485, 0.486, 0.406)
    color_std = (0.229, 0.224, 0.225) 

    train_dataset = MyDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
    val_dataset = MyDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))


    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iterator = iter(dataloader_dict["train"])
    print(batch_iterator)

    images, anno_class_images = next(batch_iterator) 


    image = images[0].numpy().transpose(1,2,0) # Convert image to numpy and exchange axis 
    plt.imshow(image)
    plt.show()

    anno_class_image = anno_class_images[0].numpy()
    plt.imshow(anno_class_image)
    plt.show()


    """
    print(len(train_img_list))
    print(len(train_anno_list))

    print(train_img_list[0])
    print(train_anno_list[0])
    """
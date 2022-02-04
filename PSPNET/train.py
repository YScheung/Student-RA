from model import PSPNet
import torch.nn as nn
import torch.nn.functional as F 
from torch import optim
import math
import torch
import torch.utils.data as data
import time
from data import make_datapath_list, MyDataset, DataTransform


# Model 
model = PSPNet(n_classes=21)

# Loss 
class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight

    def forward(self, outputs, targets): # outputs is a list 
        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

        return loss + self.aux_weight*loss_aux

criterion = PSPLoss(aux_weight=0.4) 


# Optimizer
# Customize optimizer / learning rate for each layers' parameters 
optimizer = optim.SGD([ 
    {'params': model.feature_conv.parameters(), 'lr': 1e-3}, # learning rate 1e-3
    {'params': model.feature_res_1.parameters(), 'lr': 1e-3},
    {'params': model.feature_res_2.parameters(), 'lr': 1e-3},
    {'params': model.feature_dilated_res_1.parameters(), 'lr': 1e-3},
    {'params': model.feature_dilated_res_2.parameters(), 'lr': 1e-3},
    {'params': model.pyramid_pooling.parameters(), 'lr': 1e-3},
    {'params': model.decode_feature.parameters(), 'lr': 1e-2},
    {'params': model.aux.parameters(), 'lr': 1e-2},
], momentum=0.9, weight_decay=0.0001)

# momentum: To accerlate gradients vectors / gradient descent in the right direction -> Leads to faster converging 
# https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d

# weight decay: Regularization technique by adding a small penalty
# To prevent overfitting. To keep weights small and prevent exploding gradient 
# https://medium.com/analytics-vidhya/deep-learning-basics-weight-decay-3c68eb4344e9



def lambda_epoch(epoch): 
    max_epoch = 25
    return math.pow(1-epoch/max_epoch, 0.9)


# Learning rate schedules seek to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule. 
# Common learning rate schedules include time-based decay, step decay and exponential decay.
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1) # ?


def train_model(model, dataloader_dict, criterion, scheduler, optimizer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # GPU if available, else CPU
    print("device", device)

    model.to(device)
    torch.backends.cudnn.benchmark = True # Choose the optimal training method (Hardware / GPU)

    num_train_imgs = len(dataloader_dict['train'].dataset)
    num_val_imgs = len(dataloader_dict['val'].dataset)
    batch_size = dataloader_dict['train'].batch_size

    iteration = 1

    batch_multiplier = 3


    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('Epoch {} / {}'. format(epoch+1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                scheduler.step() 
                
                # call this so that the learning rate will follow the adjusted scheme
                # Example: 
                # lr = 0.001 if epoch < 7
                # lr = 0.0001 if 7 <= epoch < 14
                # lr = 0.00001 if 14 <= epoch < 21

                optimizer.zero_grad()  

                # For every mini-batch in the training phase, we will set the gradients to zero before backpropragation 
                # When we start our training loop, we should zero out the gradients so that you do parameters will be updated correctly. 
                # Otherwise, the gradient would be a combination of the old gradient, which have already been used to update your model parameters, and the newly-computed gradient.
                #https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                
                print('train')

            else:
                if((epoch+1) % 5 == 0):
                    model.eval()
                    print('val')
                else:
                    continue
            
            count = 0
            for images, anno_class_images in dataloader_dict[phase]: # ?
                if images.size()[0] == 1:
                    continue
                images = images.to(device)
                anno_class_images = anno_class_images.to(device)

                if (phase == 'train') and (count == 0):
                    optimizer.step() # Perform parameters update
                    optimizer.zero_grad() # Reset gradient to zero 
                    count = batch_multiplier
                

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, anno_class_images.long())/batch_multiplier # PSPLoss object, to compute loss of output 

                    if phase == 'train':
                        loss.backward()

                        # loss.backward() will compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True and store them in parameter.grad attribute for every parameter.
                        # optimizer.step() updates all the parameters based on parameter.grad

                        count -= 1

                        if (iteration % 10 == 0):
                            t_iter_end = time.time()
                            duration = t_iter_end - t_iter_start 
                            print('Iteration {} || Loss: {:.6f}  || 10iter: {:.6f} sec'.format(iteration, loss.item()/batch_size*batch_multiplier, duration))

                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()*batch_multiplier
                        iteration += 1

                    else:
                        epoch_val_loss += loss.item()*batch_multiplier
        
        t_epoch_end = time.time() 
        duration = t_epoch_end - t_epoch_start  
        print('Epoch {} || Epoch_train_loss: {:.6f} || Epoch_val_loss: {:.6f}'.format(epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))        
        print('Duration {:.6f} sec'.format(duration)) # Time used for training one iteration 
        t_epoch_start = time.time() 


        torch.save(model.state_dict(), 'pspnet50_' + str(epoch) + '.pth') # Save model 


if __name__ == "__main__":

    rootpath = "./data/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_dataset = MyDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
    val_dataset = MyDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

    batch_size = 12
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # DataLoader: Combines a dataset and a sampler, and provides an iterable over the given dataset.
    # Fetches data from a dataset and serves the data up in batches


    dataloader_dict = { # Data 
        'train': train_dataloader,
        'val': val_dataloader
    }  

    num_epochs = 25
    train_model(model, dataloader_dict, criterion, scheduler, optimizer, num_epochs=num_epochs)
            

 

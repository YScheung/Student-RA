from typing import ForwardRef
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import laplace, sobel
from torch.utils.data import Dataset


# initialziation will always be switching from normal and arg sine distirbution 
# Weight initialization technique is dependent on sine actiavtion fucntions 
# Weight initialziation is always extremely important for the network to train well



def paper_init(weight, is_first=False, omega=1):  # weight initialization strategy 
    # weight: The learnable 2D weight matrix
    # is_first: Whether the linear layer is the first in the network
    # omega: Hyperparameter

    in_features = weight.shape[1]

    with torch.no_grad():  
        if is_first:
            bound = 1 / in_features
        else:
            bound = np.sqrt(6 / in_features) / omega
        
        weight.uniform_(-bound, bound)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega=30, custom_init_function=None):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if custom_init_function is None: # To use the given weight initialization strategy 
            paper_init(self.linear.weight, is_first=is_first, omega=omega)
        else:
            custom_init_function(self.linear.weight)
    
    def forward(self,x):
        return torch.sin(self.omega * self.linear(x))


class ImageSiren(nn.Module):
    def __init__(self, hidden_features, hidden_layers=1, first_omega=30, hidden_omega=30, custom_init_function=None):
        super().__init__()
        in_features = 2
        out_features = 1

        net = []
        net.append(
            SineLayer(in_features, hidden_features, is_first=True, custom_init_function=custom_init_function, omega=first_omega)
        )

        for _ in range(hidden_layers):
            net.append(
                SineLayer(hidden_features, hidden_features,is_first=False, custom_init_function=custom_init_function, omega=hidden_omega)
            )
        
        final_linear = nn.Linear(hidden_features, out_features)
        if custom_init_function is None:
            paper_init(final_linear.weight, is_first=False, omega=hidden_omega)
        else:
            custom_init_function(final_linear.weight)
        net.append(final_linear)
        self.net = nn.Sequential(*net)

    def forward(self,x):
        return self.net(x)
    




def generate_coordinates(n):
    # assume image is square

    rows, cols = np.meshgrid(range(n), range(n), indexing="ij")
    coords_abs = np.stack([rows.ravel(), cols.ravel()], axis=-1)

    return coords_abs 


class PixelDataset(Dataset): # Derivitaive of siren is siren -> Good at representing natural signals
    def __init__(self,img):
        if not (img.ndim == 2 and img.shape[0] == img.shape[1]):
            raise ValueError("Only 2D square images are supported.")
        
        self.img = img
        self.size = img.shape[0]
        self.coords_abs = generate_coordinates(self.size)
        self.grad = np.stack([sobel(img, axis=0), sobel(img, axis=1)], axis=-1) # sobel filter
        self.grad_norm = np.linalg.norm(self.grad, axis=-1)
        self.laplace = laplace(img) # 2nd derirvative 
    
    def __len__(self):
        return self.size ** 2

    def __getitem__(self, idx): # get all relevant  data for a single coordinate
        coords_abs = self.coords_abs[idx] # Extract absolute coordinate
        r, c = coords_abs # Unpack abs coordinate into row and column coordinate

        coords = 2 * ((coords_abs / self.size) - 0.5) # take abs coord and turn it into relative coord which will be in the range of -1 to 1 
        
        return {
            "coords": coords,
            "coords_abs": coords_abs,
            "intensity": self.img[r,c],
            "grad_norm": self.grad_norm[r,c],
            "grad": self.grad[r,c],
            "laplace": self.laplace[r,c],
        }


class GradientUtils:
    @staticmethod
    def gradient(target, coords):
        return torch.autograd.grad(target, coords, grad_outputs=torch.ones_like(target), create_graph=True)[0] # return gradient of the target pixel coordinate and the predicted coordinate pixel
    
    def divergence(grad, coords): # Take deriviatve of the target pixel and sum up the gradient for each pixel
        div = 0.0
        for i in range(coords.shape[1]):
            div += torch.autograd.grad( # Adding for each pixel
                grad[..., i], coords, torch.ones_like(grad[..., i]), create_graph=True)[0][..., i: i + 1] # pytorch ones like return tensor filled with 1
            
        return div 
    

    @staticmethod
    def laplace (target, coords):
        grad = GradientUtils.gradient(target, coords)
        return GradientUtils.divergence(grad, coords)


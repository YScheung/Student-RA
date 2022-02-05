import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Double Convolution layer 
class DoubleConv(nn.Module):  
    def __init__(self,in_channels, out_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels), # Batchnorm -> Bias = false
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,features=[64,128,256,512]):
        super(UNET,self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # Downsample part of UNET

        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature)) 
            in_channels = feature

        # Upsample part of UNET

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2,feature,kernel_size=2,stride=2)  # 2D Transposed convolution for upsampling (Number of channel decreases in upsampling)
            )
            self.ups.append(
                DoubleConv(feature*2, feature)
            )

        self.bottleneck = DoubleConv(features[-1], features[-1]*2) # Bottom-most layer
        self.final_conv = nn.Conv2d(features[0],out_channels, kernel_size=1) # Ouput channel is 1 ??? Output channel = number of classes ?


    def forward(self,x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # Skip connections to be appended with upsampling in the same layer to give information on 'what' 
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.ups),2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:  # Because max pool will floor the shape
                x = TF.resize(x,size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim=1) # Append conv output with upsampling output 
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


def test():
    x = torch.randn((3,1,160,160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert(preds.shape == x.shape)


if __name__ == "__main__":
    test()




        
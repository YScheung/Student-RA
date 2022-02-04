import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2DBatchNormRelu(nn.Module): 
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        output = self.relu(x)

        return output


# Extract features from images with filters. Obtain feature maps  
class FeatureMap_convolution(nn.Module):  
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()    

        # Block 1
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64 , 3, 2, 1, 1, False   # Dialation = 1: No dilation. No padding 
        self.cbnr_1 =  conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Block 2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64 , 3, 1, 1, 1, False
        self.cbnr_2 =  conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Block 3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128 , 3, 1, 1, 1, False
        self.cbnr_3 =  conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # Block 4
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self,x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        output = self.maxpool(x)

        return output


# Pass feature maps obtained from FeatureMap_Convolution block to the Residual Block. 
class ResidualBlockPSP(nn.Sequential): 
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # bottleneck PSP layer
        self.add_module("block1", bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))

        for i in range(n_blocks - 1): # Add multiple blocks
            self.add_module("block" + str(i+2), bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))



class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs


# Residual network with skip connections (Within the ResidualBlockPSP)
class bottleNeckPSP(nn.Module):  
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()

        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False) # cbr = convolution, batchnorm, relu
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)  
        self.cb_3 = conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # skip connection
        self.cb_residual = conv2DBatchNorm(in_channels,  out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True) # inplace=True: will modify the input directly, without creating new variables 

    def forward(self,x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)

        return self.relu(conv + residual)



class bottleNeckIdentifyPSP(nn.Module):  # Residual networks with no skip connections 
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()

        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False) 
        self.cb_3 = conv2DBatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x

        return self.relu(conv + residual)



# Pyramid pooling module 
# To include global + local info when determing segmentation map
# Using different types of pooling layers, it could cover the whole, half of, and small portions of the image.
# https://paperswithcode.com/method/pyramid-pooling-module

class PyramidPooling(nn.Module):  
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()

        self.height = height
        self.width = width

        out_channels = int(in_channels/len(pool_sizes))
        # pool_sizes = [6,3,2,1]
        self.avgpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avgpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation =1, bias=False)

        self.avgpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation =1, bias=False)    

        self.avgpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation =1, bias=False)

        # In Adaptive Pooling, we specify the output size. The stride and kernel-size are automatically selected to adapt to the needs.

    

    # Interpolate is needed so that output from different pooling layer will be the same and can be concantenated together 
    # Bilinear Interpolation: 
    # A resampling method that uses the distanceweighted average of the four nearest pixel values to estimate a new pixel value. '
    
    def forward(self, x):
        out1 = self.cbr_1(self.avgpool_1(x))
        out1 = F.interpolate(out1, size=(self.height, self.width), mode="bilinear", align_corners=True)

        out2 = self.cbr_2(self.avgpool_2(x))
        out2 = F.interpolate(out2, size=(self.height, self.width), mode="bilinear", align_corners=True)

        out3 = self.cbr_3(self.avgpool_3(x))
        out3 = F.interpolate(out3, size=(self.height, self.width), mode="bilinear", align_corners=True)

        out4 = self.cbr_4(self.avgpool_4(x))
        out4 = F.interpolate(out4, size=(self.height, self.width), mode="bilinear", align_corners=True)

        outputs = torch.cat([x, out1, out2, out3, out4], dim = 1) # Concatenate outputs from different pooling layers after rescaling 
        return outputs


# Decoder Module  
class DecodePSPFeature(nn.Module):  
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        self.height = height
        self.width = width

        # 1x1 filter to reduce number of feature maps 
        # https://machinelearningmastery.com/introduction-to-1x1-convolutions-to-reduce-the-complexity-of-convolutional-neural-networks/
        
        self.cbr = conv2DBatchNormRelu(in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False) 
        self.dropout = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        outputs = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True) # Upsample, final output, final image
        
        return outputs


# Auxiliary loss: To reduce the vanishing gradient problem + stabilizes the training, used regularization. 
# Usually used for very deep networks (Resnet)
# It's only used for training and not for inference.
# It's used for Resnet in feature map extraction in this paper 
# https://stats.stackexchange.com/questions/304699/what-is-auxiliary-loss-as-mentioned-in-pspnet-paper

class AuxiliaryPSPLayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super(AuxiliaryPSPLayers, self).__init__()

        # forward
        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode="bilinear", align_corners=True) # Align corners = True is a common practice that will yield better results. WHY ?

        return output


class PSPNet(nn.Module):   
    def __init__(self, n_classes):
        super(PSPNet,self).__init__()

        # parameters
        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60

        # Feature extraction module 
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256, stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP(n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512, stride=2, dilation=1)
        self.feature_dilated_res_1 = ResidualBlockPSP(n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP(n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048, stride=1, dilation=4)

        # Pyramid Pooling module 
        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6,3,2,1], height=img_size_8, width=img_size_8)

        # Decoder Module (Upsampling)
        self.decode_feature = DecodePSPFeature(height=img_size, width=img_size, n_classes=n_classes)

        # Auxilary Loss Module 
        self.aux = AuxiliaryPSPLayers(in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

    
    def forward(self,x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)
        output_aux = self.aux(x)
        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return (output, output_aux)





if __name__ == "__main__":
   # x = torch.randn(1, 3, 475, 475)
    #feature_conv = FeatureMap_convolution()
    #outputs = feature_conv(x)
    #print(outputs.shape)
    
    dummy_img = torch.rand(2, 3, 475, 475) # 2 images, RGB channels, 475x475
    net = PSPNet(21)
    outputs = net(dummy_img)
    print(outputs[0].shape)
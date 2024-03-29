'''
Start code for Project 3
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Piyush Nagasubramaniam
    PSU Email ID: pvn5119@psu.edu
    Description: (A short description of what each of the functions you've written does.).
}
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet101, ResNet101_Weights, 
    densenet121, DenseNet121_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights
)
from torch import Tensor

# TODO: Can the MLP be improved?
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): number of input features
            hidden_dim (int): number of hidden units
            output_dim (int): number of output units
        """
        super(MLP, self).__init__()

        # Baseline Architecture
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Linear(hidden_dim, output_dim),
        # )
        
        # Modified Arch 1
        # Best Hyperparameters: (Batch size=1024, lr=0.010, epochs=20)
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim//2),

        #     nn.Linear(hidden_dim//2, hidden_dim//2),
        #     nn.ReLU(),

        #     nn.Linear(hidden_dim//2, hidden_dim//2),
        #     nn.ReLU(),

        #     nn.Linear(hidden_dim//2, output_dim),
        # )

        # Modified Arch 2
        # Best Hyperparameters: (Batch size=1024, lr=0.016, epochs=20)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),

            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2, eps=1e-7, momentum=0.1),
            nn.ReLU6(),

            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2, eps=1e-7, momentum=0.1),
            nn.ReLU6(),

            nn.Linear(hidden_dim//2, output_dim),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return F.log_softmax(self.layers(x), dim=1)

class CNN(nn.Module):
    # Baseline Arch Best Hyperparameters: (Batch size: 1024, lr: 0.002)
    # Modified Arch 1 Best Hyperparameters: (Batch size: 1024, lr: 0.008, Epochs: 40)
    # Modified Arch 2 Best Hyperparameters: (Batch size: 512, lr: 0.004, Epochs: 20)
    def __init__(self, input_channels=1, img_size=32, num_classes=17):
        """
        Args:
            input_channels (int): number of channels in the input image
            img_size (int): size of the input image (img_size x img_size)
            num_classes (int): number of classes in the dataset
        """
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Baseline Architecture
        self.conv_layers= nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.fc_1 = nn.Linear(128 * (self.img_size // 8) * (self.img_size // 8), 1024)
        self.fc_2 = nn.Linear(1024, self.num_classes)

        # Modified Arch 1
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(),

        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(),

        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
            
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.LeakyReLU(),
        # )
        # self.fc_1 = nn.Linear(128 * (self.img_size // 8) * (self.img_size // 8), 1024)
        # self.fc_2 = nn.Linear(1024, self.num_classes)

        # Modified Arch 2
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(),

        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(),

        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(128, 64, kernel_size=1),
        #     nn.BatchNorm2d(64, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.LeakyReLU(),

        #     nn.Conv2d(64, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
            
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.LeakyReLU(),
        # )
        # self.fc_1 = nn.Linear(16 * (self.img_size // 8) * (self.img_size // 8), 1024)
        # self.fc_2 = nn.Linear(1024, self.num_classes)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)

# TODO: Can the CNN be improved? You may want to add or remove any arguments to the init.
class CNN2(nn.Module):
    def __init__(self, input_channels=1, img_size=32, num_classes=17):
        """
        Args:
            input_channels (int): number of channels in the input image
            img_size (int): size of the input image (img_size x img_size)
            num_classes (int): number of classes in the dataset
        """
        super(CNN2, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.img_size = img_size

        # Modified CNN Architecture 2
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(),

            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, eps=1e-7, momentum=0.1, affine=True, track_running_stats=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )
        self.fc_1 = nn.Linear(16 * (self.img_size // 8) * (self.img_size // 8), 1024)
        self.fc_2 = nn.Linear(1024, self.num_classes)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return F.log_softmax(x, dim=1)


class Resnet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(Resnet, self).__init__()
        model = resnet101()
    
        if pretrained:
            model.load_state_dict( ResNet101_Weights.IMAGENET1K_V2.get_state_dict(progress=True) )
        
        self.out_channels = model.fc.in_features
        self.backbone = nn.Sequential( *list(model.children())[:-1] )
        last = nn.Linear(self.out_channels, 17)
        last.apply(self._he_init)
        self.last = last
    

    def _he_init(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = x.view(-1, self.out_channels)
        x = self.last(x)
        return F.log_softmax(x, dim=1)
    

class Densenet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super(Densenet, self).__init__()
        model = densenet121()

        if pretrained:
            state_dict = DenseNet121_Weights.DEFAULT.get_state_dict(progress=True)
            for key in list(state_dict.keys()):
                state_dict[key.replace('.1.', '1.'). replace('.2.', '2.')] = state_dict.pop(key)
            model.load_state_dict(state_dict) 
        
        self.out_channels = model.classifier.in_features
        self.backbone = nn.Sequential( *list(model.children())[:-1] )
        avg_pool = nn.AdaptiveAvgPool2d((1,1))
        last = nn.Linear(self.out_channels, 17)

        avg_pool.apply(self._xavier_init)
        last.apply(self._xavier_init)
        self.avg_pool = avg_pool
        self.last = last
    

    def _xavier_init(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = x.view(-1, self.out_channels)
        x = self.last(x)
        return F.log_softmax(x, dim=1)


class Mobilenet(nn.Module):
    def __init__(self, pretrained:bool = False):
        super(Mobilenet, self).__init__()
        model = mobilenet_v3_small()

        if pretrained:
            state_dict = MobileNet_V3_Small_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
            model.load_state_dict(state_dict)
        
        self.out_channels = model.classifier[0].in_features
        self.backbone = nn.Sequential( *list(model.children())[:-1] )
        classifier = nn.Linear(self.out_channels, 17)
        classifier.apply(self._he_init_)
        self.classifier = classifier
    

    def _he_init_(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    
    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = x.view(-1, self.out_channels)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
        
        

if __name__ == '__main__':
    model = Mobilenet()
    trainable = sum([ p.numel() for p in model.parameters() if p.requires_grad ])
    print("Number of trainable parameters in Mobilenet Small (full): {}".format(trainable))

    del model
    model = Resnet(pretrained=True)
    for child in list( model.children() )[0][:-1][:-2]:
        for param in child.parameters():
            param.requires_grad_(False)

    for param in list(model.children())[0][:-1][:-1][-1][:-5].parameters():
        param.requires_grad_(False)
    
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("Number of trainable parameters in Resnet (finetuning): {}".format(trainable))

    del model
    model = Densenet(pretrained=True)
    for child in list(model.children())[:-2][0][0][:-2]:
        for param in child.parameters():
            param.requires_grad_(False)
    
    last_dense_block = list(model.children())[:-2][0][0][:-1][-1]
    freeze_layers = [
        last_dense_block.denselayer1, last_dense_block.denselayer2, 
        last_dense_block.denselayer3, last_dense_block.denselayer4,
        last_dense_block.denselayer5, last_dense_block.denselayer6,
        last_dense_block.denselayer7, last_dense_block.denselayer8, 
        last_dense_block.denselayer9, last_dense_block.denselayer10,
        last_dense_block.denselayer11, last_dense_block.denselayer12,
        last_dense_block.denselayer13,
        # Less tuning:
        last_dense_block.denselayer14, last_dense_block.denselayer15
    ]
    for layer in freeze_layers:
        for param in list( layer.parameters() ):
            param.requires_grad_(False)
    
    trainable = sum( [p.numel() for p in model.parameters()] )
    print( "Number of trainable parameters in Densenet (finetuning): {}".format(trainable) )
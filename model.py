import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from utilsTorch import *


class YOLOv1(torch.nn.Module):
    def __init__(self,model = "mobilenet"):

        super().__init__()
        
        if model == "resnet":
            resnet = torchvision.models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
            
            self.conv = nn.Sequential(
            nn.Conv2d(2048,1024,3),
            nn.Conv2d(1024,1024,3,stride=2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024,1024,3),
            nn.Conv2d(1024,1024,3),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d(1)
            )
            
        elif model == "mobilenet":
            mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
            
            self.conv = nn.Sequential(
            nn.Conv2d(1280,1024,3),
            nn.Conv2d(1024,1024,3,stride=2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024,1024,3),
            nn.Conv2d(1024,1024,3),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d(1)
            )
                         
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
                
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024,4096)
        self.predictor = nn.Linear(4096,7*7*30)
        
        self.leaky = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        features = self.feature_extractor(x)
        z = self.flatten(self.conv(features))
        z = self.leaky(self.fc1(z))
        predictions = self.sigmoid(self.predictor(z))
         
        return predictions

    
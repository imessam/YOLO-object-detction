import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from utilsTorch import *


class YOLOv1(torch.nn.Module):
    def __init__(self):

        super().__init__()
        #resnet = torchvision.models.resnet18(pretrained=True)
        #densenet = torchvision.models.densenet121(pretrained=True)
        mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        
        
        self.conv = nn.Sequential(
            nn.Conv2d(1280,1024,3),
            nn.Conv2d(1024,1024,3,stride=2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024,1024,3),
            nn.Conv2d(1024,1024,3),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d(1)
            )
        
        
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

    
def yoloLossFaster(labels,predictions,lambda_coords,lambda_noobj):
    
    
    N,R,C,D = labels.shape
    
    real_bbox = labels[:,:,:,:10].view((N,R,C,2,5))
    pred_bbox = predictions[:,:,:,:10].view((N,R,C,2,5))
    #print(real_bbox.shape,pred_bbox.shape)
    
    
    real_classes = labels[:,:,:,10:]
    pred_classes = predictions[:,:,:,10:]
    #print(real_classes.shape,pred_classes.shape)
    
    real_conf = real_bbox[:,:,:,:,0]
    pred_conf = pred_bbox[:,:,:,:,0]
    #print(real_conf.shape,pred_conf.shape)
    
    real_xy = real_bbox[:,:,:,:,1:3]
    pred_xy = pred_bbox[:,:,:,:,1:3]
    #print(real_xy.shape,pred_xy.shape)
    
    real_wh = real_bbox[:,:,:,:,3:]
    pred_wh = pred_bbox[:,:,:,:,3:]
    #print(real_wh.shape,pred_wh.shape)

    
    conf_loss = 0
    xy_loss = 0
    wh_loss = 0
    classes_loss = 0
    
    xy_loss = torch.sum(((real_xy - pred_xy)**2)*real_conf.unsqueeze(4))*lambda_coords
    wh_loss = torch.sum(((torch.sqrt(real_wh) - torch.sqrt(pred_wh))**2)*real_conf.unsqueeze(4))*lambda_coords

    classes_loss = torch.sum(((real_classes - pred_classes)**2)*torch.max(real_conf,dim=3)[0].unsqueeze(3))

    iou = iou_grid([pred_xy,pred_wh],[real_xy,real_wh])
    
    conf_loss = torch.sum(((real_conf*iou-pred_conf)**2)*real_conf) + torch.sum(((real_conf*iou-pred_conf)**2)*(1-real_conf))*lambda_noobj
    
    
    
    #print(iou,conf_loss,classes_loss,xy_loss,wh_loss)                
    return (conf_loss+classes_loss+xy_loss+wh_loss)
    
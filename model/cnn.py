'''
Aum Sri Sai Ram
                
Authors: Darshan Gera 
Date: 20-06-2022
Email: darshangera@sssihl.edu.in
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from model.resnet import *
import pickle
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
def resModel(args): #resnet18
    
    model = torch.nn.DataParallel(resnet18(end2end= False,  pretrained= False)).to(device)
    
    if args.pretrained:
       
       checkpoint = torch.load('pretrained/res18_naive.pth_MSceleb.tar')
       pretrained_state_dict = checkpoint['state_dict']
       model_state_dict = model.state_dict()
       
       '''
       for name, param in pretrained_state_dict.items():
           print(name)
           
       for name, param in model_state_dict.items():
           print(name)
       '''  
       for key in pretrained_state_dict:
           if  ((key == 'module.fc.weight') | (key=='module.fc.bias') | (key=='module.feature.weight') | (key=='module.feature.bias') ) :
               print(key) 
               pass
           else:
               #print(key)
               model_state_dict[key] = pretrained_state_dict[key]

       model.load_state_dict(model_state_dict, strict = False)
       print('Model loaded from Msceleb pretrained')
    else:
       print('No pretrained resent18 model built.')
    return model   


'''
Aum Sri Sai Ram

Email: darshangera@sssihl.edu.in
'''

# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from model.cnn import resModel,resModel_50
import numpy as np
from common.utils import accuracy
import os
from algorithm.loss import * 



class noisyfer:
    def __init__(self, args, train_dataset, device, input_channel, num_classes):

        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr
        
        self.relabel_epochs = args.relabel_epochs       
        self.margin = args.margin
        self.relabled_count = 0
        self.eps = args.eps
        self.warmup_epochs = args.warmup_epochs
        self.alpha = args.alpha 
        
        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq        
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset
        self.co_lambda_max = args.co_lambda_max
        self.beta = args.beta
        self.num_classes  = args.num_classes
        self.max_epochs = args.n_epoch

        if  args.model_type=="res":               
            self.model = resModel(args)
            

        self.model = self.model.to(device)
        
        
        self.weighted_CCE =  DCE(num_class=args.num_classes, reduction='mean')
        
        filter_list = ['module.classifier.weight', 'module.classifier.bias']
        
        base_parameters_model = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in filter_list, self.model.named_parameters()))))
        
       
        self.optimizer = torch.optim.Adam([{'params': base_parameters_model}, {'params': list(self.model.module.classifier.parameters()), 'lr': learning_rate}], lr=1e-3)
                 
                                             
        print('\n Initial learning rate is:')
        for param_group in self.optimizer.param_groups:
            print(  param_group['lr'])                              
        
        if args.resume:
           
           pretrained = torch.load(args.resume)
           pretrained_state_dict1 = pretrained['model']   
            
           model_state_dict =  self.model.state_dict()
           loaded_keys = 0
           total_keys = 0
           for key in pretrained_state_dict1:                
               if  ((key=='module.fcx.weight')|(key=='module.fcx.bias')):
                   print(key)
                   pass
               else:    
                   model_state_dict[key] = pretrained_state_dict1[key]
                   total_keys+=1
                   if key in model_state_dict :
                      loaded_keys+=1
           print("Loaded params num:", loaded_keys)
           print("Total params num:", total_keys)
           self.model.load_state_dict(model_state_dict) 
            
           print('Model loaded from ',args.resume)
           
        else:
           print('\n No checkpoint found.\n')         
        
        self.ce_loss = torch.nn.CrossEntropyLoss().to(device)
        
        self.m1_statedict =  self.model.state_dict()
        self.o_statedict = self.optimizer.state_dict()  

        self.adjust_lr = args.adjust_lr
    
    
        
    # Evaluate the Model
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model.eval()  
        correct1 = 0
        total1 = 0
        correct  = 0
        with torch.no_grad():
            for images,_, labels, _ in test_loader:
                images = (images).to(self.device)
                logits1 = self.model(images)
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels).sum()
            
                 
                _, avg_pred = torch.max(outputs1, 1)
                correct += (avg_pred.cpu() == labels).sum()
                
            acc1 = 100 * float(correct1) / float(total1)
           
           
        return acc1
       
    def save_model(self, epoch, acc, noise):
        torch.save({' epoch':  epoch,
                    'model': self.m1_statedict,
                    'optimizer':self.o_statedict,},                          
                     os.path.join('checkpoints/', "epoch_"+str(epoch)+'_noise_'+noise+"_acc_"+str(acc)[:5]+".pth")) 
        print('Models saved '+os.path.join('checkpoints/', "epoch_"+str(epoch)+'_noise_'+noise+"_acc_"+str(acc)[:5]+".pth")) 
    
    
    
               
    # Train the Model
    def train(self, train_loader, epoch):
        print('Training ...')
        self.model.train() 
        
        
        if epoch > 0:
           self.adjust_learning_rate(self.optimizer, epoch)
        
        train_total = 0
        train_correct = 0
        
        pure_ratio_1_list = []
        pure_ratio_2_list = []
        
        
        
        
        if epoch < self.warmup_epochs:
            print('\n Warm up stage using supervision loss based on easy samples')
        elif epoch == self.warmup_epochs:
            print('\n Robust learning stage using consistency loss combined with supervision loss based on selected clean samples based on dynamic threshold')
        
        #all_indices = []    
        all_avg_thresholds= []
        
        all_clean_len_list = []
        all_noise_len_list = []
        epoch_avg_post =torch.zeros(self.num_classes,1,requires_grad=False).float().cuda()
        clean_count = 0
        
        
        
        for i, (images1, images2, labels, indexes) in enumerate(train_loader):
        
            ind = indexes.cpu().numpy().transpose()
        
            
            images1 = images1.to(self.device)
            images2 = images2.to(self.device)
            labels = labels.to(self.device)
            # Forward + Backward + Optimize
            
            logits1 = self.model(images1)
            logits2 = self.model(images2)
            
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            
            
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1
            
            
            if epoch < self.warmup_epochs:               
               loss = (self.ce_loss(logits1,labels) + self.ce_loss(logits2, labels))/2.0
               
               probs = F.softmax(logits1, dim=1)
               avg = torch.mean(probs , dim=0)
               avg1 = avg.reshape(-1, 1)
               
              
              
               
            else:
               loss1, clean_indices1, noisy_indices1,avg1 = self.weighted_CCE(logits1,labels)
               loss2, clean_indices2, noisy_indices2,avg2 = self.weighted_CCE(logits2,labels)
               
               loss_c = (loss1 + loss2)/2.0 #Superivison loss
               
               all_clean_len_list.append(clean_indices1) 
               
               loss_o = symmetric_kl_div(probs1, probs2).mean() #consistency loss
               
               loss = (1 - self.alpha) * loss_c + loss_o * self.alpha
                  
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Loss1: %.4f'
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1, loss.data.item() ))

        train_acc1 = float(train_correct) / float(train_total)
        
        
        
        return train_acc1
            
            
            
    
    def adjust_learning_rate(self, optimizer, epoch):
        print('\n******************************\n\tAdjusted learning rate: '+str(epoch) +'\n')    
        for param_group in optimizer.param_groups:
           param_group['lr'] *= 0.95
           print(param_group['lr'])              
        print('******************************')
    

    
    
    
    
    
    

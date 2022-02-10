import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.init as init


class DeepNeuralNetwork(nn.Module):

    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()


        self.feature = nn.Sequential(
            nn.MaxPool1d(6, stride=2, ceil_mode=True),
            #nn.Dropout(0.5),
            #nn.MaxPool1d(3, stride=2, ceil_mode=True)
        )
        
        self.main = nn.Sequential(
                nn.Linear(in_features=4284,  out_features=1024), # 4284
                #nn.BatchNorm1d(512, eps=0.001, momentum=0.1),
                #nn.ReLU(),
                #nn.BatchNorm1d(1024, eps=0.001, momentum=0.1),
                #nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                nn.ReLU(),
                nn.Dropout(0.7), 
                
                nn.Linear(in_features=1024, out_features=512),
                #nn.BatchNorm1d(256, eps=0.001, momentum=0.1),
                #nn.ReLU(),
                #nn.BatchNorm1d(512, eps=0.001, momentum=0.1),
                #nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                nn.ReLU(),    
                nn.Dropout(0.7), 
            
                nn.Linear(in_features=512, out_features=128),
                #nn.BatchNorm1d(128, eps=0.001, momentum=0.1),
                #nn.ReLU(),
                #nn.BatchNorm1d(128, eps=0.001, momentum=0.1),
                #nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
                nn.ReLU(),
                nn.Dropout(0.7),
    
        )

        self.logits = nn.Linear(in_features=128, out_features=2)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1) 
        
        self.init_weight(self.main)

    def forward(self, x):
        #x_swp = x.view(batch_size, time, feature)
        x = self.feature(x)
        
        size       = x.shape
        batch_size = size[0]
        time       = size[1]
        feature    = size[2]

        flatten = x.reshape(batch_size, time*feature)
        out = self.main(flatten)
        logits = self.logits(out)        
        softmax = self.softmax(logits)
        logsoftmax = self.logsoftmax(logits) 
        uncertanity, prob, alpha = self.compute_uncertanity(
            relu_evidence=self.exp_evidence, logits=logits)
            
        return logits, logsoftmax, softmax, uncertanity, prob, alpha 

    def relu_evidence(self):
        return F.relu
    
    def exp_evidence_2(self):
        return torch.exp

    def exp_evidence(self, logits): 
        return torch.exp(torch.clamp(logits,-10,10))
    
    def softplus_evidence(self):
        return F.softplus

    def compute_uncertanity(self, relu_evidence, logits):
        K        = 2
        evidence = relu_evidence(logits)
        alpha    = evidence + 1
        uncrtnty = K / torch.sum(alpha, dim=1) 
        prob     = alpha / torch.sum(alpha, dim=1).reshape(-1,1)
        return uncrtnty, prob, alpha
    
    def init_weight(self, m):
        for each_module in m:
            if "Linear" in each_module.__class__.__name__:
                init.xavier_normal(each_module.weight)
                init.constant(each_module.bias, 0.)




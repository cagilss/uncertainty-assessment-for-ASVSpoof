import torch
import math
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

class DeepNeuralNetwork(nn.Module):

    def __init__(self): 
        super(DeepNeuralNetwork, self).__init__()

        # use LSTM as feature axtractor 
        self.lstm = nn.LSTM(input_size=72, hidden_size=16, num_layers=10, batch_first=True, bidirectional=False, dropout=0)  

        # dropout 
        # batch norm
        # dropout

        # try 30 epoch, it semms a better option not to overfit dataset

        self.main = nn.Sequential(
                nn.Linear(in_features=2016,  out_features=512), # 2016
                #nn.BatchNorm1d(512, eps=0.001, momentum=0.1),
                nn.ReLU(),
                nn.BatchNorm1d(512, eps=0.001, momentum=0.1),
                #nn.Dropout(0.5),

                # instead of using dropout we can use maxpool for feature extractor
                # instead of randomly cancels layers we can just take the max 

                nn.Linear(in_features=512, out_features=256),
                #nn.BatchNorm1d(256, eps=0.001, momentum=0.1),
                nn.ReLU(),
                nn.BatchNorm1d(256, eps=0.001, momentum=0.1),
                #nn.Dropout(0.5),
                
                nn.Linear(in_features=256, out_features=128),
                #nn.BatchNorm1d(128, eps=0.001, momentum=0.1),
                nn.ReLU(),
                nn.BatchNorm1d(128, eps=0.001, momentum=0.1),
                #nn.Dropout(0.5),

                #nn.Linear(in_features=128, out_features=1024),
                #nn.BatchNorm1d(1024),
                #nn.ReLU(),
                #nn.Dropout(0.5),
                
                #nn.Linear(in_features=1024, out_features=1024),
                #nn.BatchNorm1d(1024),
                #nn.ReLU(),
                #nn.Dropout(0.5),
                )      

        self.logits = nn.Linear(in_features=128, out_features=2)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1) 

        self.init_weight(self.main)

    def forward(self, x):
        hidden_size = 16
        size = x.shape
        batch_size = size[0]
        time = size[1]
        feat = size[2]

        out, hn = self.lstm(x)
        #x_re = x.view(batch_size, time*feat)
        flatten = out.reshape(batch_size, time*hidden_size)

        out = self.main(flatten)

        logits = self.logits(out)
        #sigmoid_logits = self.sigmoid(logits)

        logsoftmax = self.logsoftmax(logits)
        softmax = self.softmax(logits)
        
        uncertanity, prob, alpha = self.compute_uncertanity(
            relu_evidence=self.exp_evidence(), logits=logits)

        return logits, logsoftmax, softmax, uncertanity, prob, alpha

    def compute_uncertanity(self, relu_evidence, logits):
        K        = 1
        evidence = relu_evidence(logits)
        alpha    = evidence + 1
        uncrtnty = K / torch.sum(alpha, dim=1) 
        prob     = alpha / torch.sum(alpha, dim=1).reshape(-1,1)
        return uncrtnty, prob, alpha

    def relu_evidence(self):
        return F.relu
    
    def exp_evidence(self):
        return torch.exp
    
    def softplus_evidence(self):
        return F.softplus

    def init_weight(self, m):
        for each_module in m:
            if "Linear" in each_module.__class__.__name__:
                init.xavier_normal(each_module.weight)
                init.constant(each_module.bias, 0.)


class Bidirectional_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        self.b_lstm = nn.LSTM(input_size=72, hidden_size=self.hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        hidden_size_vector = (Variable(torch.zeros(2, 1, self.hidden_size)), 
                        Variable(torch.zeros(2, 1, self.hidden_size)))
        out, hn = self.b_lstm(x, hidden_size_vector)
        return out 
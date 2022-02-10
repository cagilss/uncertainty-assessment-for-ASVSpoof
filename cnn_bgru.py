import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable



class Convolutional_BGRU(nn.Module):
    def __init__(self):
        super(Convolutional_BGRU, self).__init__()
        self.conv_1    = conv_layer(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batch_n_1 = nn.BatchNorm2d(num_features=32, eps=0.1, momentum=0.1)
        self.relu_1    = nn.LeakyReLU(0.01)
        self.dropout_1 = nn.Dropout(0.5)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=(2,1), padding=0)

        self.conv_2    = conv_layer(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=2)
        self.batch_n_2 = nn.BatchNorm2d(num_features=32, eps=0.1, momentum=0.1)
        self.relu_2    = nn.LeakyReLU(0.01)
        self.dropout_2 = nn.Dropout(0.5)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=(2,1), padding=0)
        #self.conv_3 = conv_layer(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        #self.relu_3    = nn.ReLU()
        
        self.conv_3    = conv_layer(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=2)
        self.batch_n_3 = nn.BatchNorm2d(num_features=32, eps=0.1, momentum=0.1)
        self.relu_3    = nn.LeakyReLU(0.01)
        self.dropout_3 = nn.Dropout(0.5)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=(2,1), padding=0)
        
        self.conv_4    = conv_layer(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.batch_n_4 = nn.BatchNorm2d(num_features=32, eps=0.1, momentum=0.1)
        self.relu_4    = nn.LeakyReLU(0.01)
        self.dropout_4 = nn.Dropout(0.5)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=(2,1), padding=0)

        self.bgru      = Bidirectional_GRU(input_size=132, hidden_size=16, num_layers=1)

        self.fc_7      = fully_connected(in_features=1024, out_features=512)
        self.relu_7    = nn.LeakyReLU(0.01)
        self.fc_8      = fully_connected(in_features=512, out_features=256)
        self.relu_8    = nn.LeakyReLU(0.01)
        self.logits    = fully_connected(in_features=256, out_features=2)

        self.softmax_logits = nn.Softmax(dim=1)
        self.sigmoid_logits = nn.Sigmoid()

        self.init_weight()

    def forward(self, x):
        size       = x.shape
        batch_size = size[0]
        feature    = size[1]
        time       = size[2]

        x_swp = x.view(batch_size, time, feature)
        x_exp = x_swp.view(batch_size, -1, time, feature)

        conv_1    = self.conv_1(x_exp)
        batch_1 = self.batch_n_1(conv_1)
        relu_1    = self.relu_1(batch_1)
        drop_1 = self.dropout_1(relu_1)
        maxpool_1 = self.maxpool_1(drop_1)

        conv_2    = self.conv_2(maxpool_1)
        batch_2 = self.batch_n_2(conv_2)
        relu_2    = self.relu_2(batch_2)
        drop_2 = self.dropout_2(relu_2)
        maxpool_2 = self.maxpool_3(drop_2)

        conv_3    = self.conv_3(maxpool_2)
        batch_3 = self.batch_n_3(conv_3)
        relu_3    = self.relu_3(batch_3)
        drop_3 = self.dropout_3(relu_3)
        maxpool_3 = self.maxpool_3(drop_3)

        conv_4    = self.conv_4(maxpool_3)
        batch_4 = self.batch_n_3(conv_4)
        relu_4    = self.relu_4(batch_4)
        drop_4 = self.dropout_4(relu_4)
        maxpool_4 = self.maxpool_3(drop_4)

        bdr_gru   = self.bgru(maxpool_4)

        flatten   = bdr_gru.view(batch_size, 16*32*2)

        fc_7      = self.fc_7(flatten) 
        relu_7    = self.relu_7(fc_7)
        fc_8      = self.fc_8(relu_7)
        relu_8    = self.relu_7(fc_8)
        logits    = self.logits(relu_8)
        
        softmax_logits = self.softmax_logits(logits)
        sigmoid_logits = self.sigmoid_logits(logits)

        #sigmoid_logits_log  = torch.log(sigmoid_logits)
        #sig_logits_log_mean = torch.mean(sigmoid_logits_log)

        # reduce_mean(sigmoid_logits_log)
        uncertanity = self.compute_uncertanity(
            relu_evidence=F.relu, logits=logits)

        return logits

    def compute_uncertanity(self, relu_evidence, logits):
        K        = 2
        evidence = relu_evidence(logits)
        alpha    = evidence + 1
        uncrtnty = K / torch.sum(alpha, dim=1) 
        # prob     = alpha / torch.sum(alpha, dim=1) 
        return uncrtnty
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.zero_()

    def size_info(self, x):
        size = x.shape
        return size[0], size[1], size[2], size[3]


class Bidirectional_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Bidirectional_GRU, self).__init__()
        self.bi_gru = nn.GRU(input_size=input_size, 
        hidden_size=hidden_size, num_layers=num_layers, 
        batch_first=True, bidirectional=True)

    def forward(self, x):
        List         = []
        channel_size = x.shape[1]
        for c in range(0, channel_size):
            per_chn = x[:, c, :, :]
            out, hn = self.bi_gru(per_chn)
            hn_size = hn.shape
            hn_ex   = hn.view(hn_size[1], hn_size[0], hn_size[2], 1)
            List.append(hn_ex) 
        return torch.cat(List, dim=3)


class conv_layer_mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
    stride, padding):
        super(conv_layer_mfm, self).__init__()
        self.out_channels = out_channels
        self.conv_layer = nn.Conv2d(in_channels, 2*out_channels, 
        kernel_size, stride, padding)

    def forward(self, x):
        conv = self.conv_layer(x)
        out  = torch.split(conv, self.out_channels, 1)
        return torch.max(out[0], out[1])


class conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
    stride, padding):
        super(conv_layer, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, 
        kernel_size, stride, padding)

    def forward(self, x):
        conv = self.conv_layer(x)
        return conv


class fully_connected_mfm(nn.Module):
    def __init__(self, in_features, out_features, MFM=True):
        super(fully_connected_mfm, self).__init__()
        self.MFM = MFM
        self.out_features = out_features 
        if MFM:
            self.fully_connected = nn.Linear(in_features, 2*out_features)
        else:
            self.fully_connected = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.fully_connected(x)
        if self.MFM:
            o_s = torch.split(out, self.out_features, 1)
            out = torch.max(o_s[0], o_s[1])
        return out
    

class fully_connected(nn.Module):
    def __init__(self, in_features, out_features):
        super(fully_connected, self).__init__()
        self.fully_connected = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.fully_connected(x)
        return out
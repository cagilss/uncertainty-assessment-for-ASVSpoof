import sys
import os
import platform
import random 

from evaluation import Evaluation 
import datetime
from torch.optim import Adam, ASGD, SGD 
from batch_window_size_adjuster import BatchWindowSizeAdjuster
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import librosa
from torch import Tensor
# from torchvision import transforms
from joblib import Parallel, delayed

from models import LSTMNeuralNetwork, DeepNeuralNetwork, Convolutional_BGRU, LCNN, LightResNet
# from asv_dataset import ASVDataset
from load_dataset import LoadDataset
from pre_processor import PreProcessor
from tensorboardX import SummaryWriter 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from profilehooks import profile
from load_dataset import ASVDatasetTorch, asvdataset_collate_fn_pad, BinnedLengthSampler

import torch_ard as nn_ard


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


# global step counter
class GlobalStep:
    def __init__(self):
        self.global_step = 0
    
    def set_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return self.global_step


def run_model(data, args, delete_catches=False):
    run_session(data, args) 


def get_current_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def init_model(model, cuda, typ):
    if not cuda:
        raise ValueError('cuda is not available')
    
    if model == 'CNN_BGRU':
        return Convolutional_BGRU().to(device)
    if model == 'LSTM':
        return LSTMNeuralNetwork().to(device)
    if model == 'DNN':
        return DeepNeuralNetwork().to(device)
    if model == 'LCNN':
        return LCNN().to(device)
    if model == 'LightResNet':
        return LightResNet().to(device)



def init_loss(loss, cuda):
    if not cuda:
        raise ValueError('cuda is not available')
    if loss == 'cross_entropy':
        return nn.CrossEntropyLoss().to(device)
    if loss == 'binary_cross_entropy':    
        return nn.BCELoss()
    if loss == 'binary_cross_entropy_with_sigm':
        pos_weight = torch.ones([1]).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss == 'NLLoss':
        weight = torch.FloatTensor([1.0, 9.0])
        return nn.NLLLoss(weight=weight)


def init_optimizer(optimizer, model, lr=0.001):
    if model is None:
        raise ValueError('Model is not initialized')
    if optimizer == 'adam':
        return Adam(
                    model.parameters(), 
                    lr=lr, 
                    eps=0.1
                    )
    if optimizer == 'ASGD':
        return ASGD(model.parameters(), 
                    lr=2e-5, 
                    weight_decay=5e-2
                    ) 
    if optimizer == 'SGD':
        return SGD(model.parameters(), lr=lr)


def KL(alpha):
    K = 2
    #beta = tf.constant(np.ones((1,K)))
    beta = torch.FloatTensor(np.ones((1, K))).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - \
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, 
                    keepdim=True) + lnB + lnB_uni.to(device)
    return kl


def mse_loss(p, alpha, global_step, annealing_step): 
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S

    A = torch.sum((p-m)**2, dim=1, keepdim=True)
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), dim=1, keepdim=True)

    annealing_coef = min(1.0, (global_step/annealing_step))
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp)
    return (A + B) + C


def loss_EDL(func=torch.digamma):
    def loss_func(p, alpha, global_step, annealing_step): 
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        A = torch.sum(p * (func(S) - func(alpha)), dim=1, keepdim=True)
        annealing_coef = min(1.0, (global_step/annealing_step))
        alp = E*(1-p) + 1 
        B =  annealing_coef * KL(alp)
        return (A + B)
    return loss_func


def one_hot_encoding(label):
    List = torch.LongTensor(np.zeros((label.shape[0], 2)))
    for idx,L in enumerate(label):
        List[idx][L] = 1
    return List

def run_session(data, args):    
    #tr_dataloader, val_dataloader, eval_dataloader = data[0], data[1], data[2]
    tr_dataloader, eval_dataloader = data[0], data[1]

    g         = GlobalStep() # initialize global step
    epoch     = args.ep
    cuda      = torch.cuda.is_available()
    curr_time = get_current_time()
    model     = init_model(args.model, cuda, typ=1)
    loss      = init_loss('cross_entropy', cuda)
    #loss      = loss_EDL(torch.digamma) # custom loss algorithm
    #loss      = mse_loss # custom loss
    # if logits output is 2d use cross_entropy loss function
    optimizer = init_optimizer('adam', model, lr=args.lr)
    
    if args.db:
        print('Running the model')

    #train and test the model     
    for e in range(epoch):        
        if args.trModel:
            train2(
                model=model,
                loss_func=loss,
                optimizer=optimizer,
                g=g, # global step
                tr_dataloader=tr_dataloader,
                cuda=cuda,
                e=e
                )
        if args.tsModel:
            test(
                model=model,
                loss_func=loss,
                optimizer=optimizer,
                g=g, # globbal step
                cuda=cuda,
                data=data,
                e=e,
                args=args
                )


def test(model, loss_func, optimizer, g, cuda, e, data, args):
    # test_(model, loss_func, optimizer, g, data[0], cuda, e, mode='train',args=args)
    # test_(model, loss_func, optimizer, g, data[1], cuda, e, mode='dev', args=args)
    # test_(model, loss_func, optimizer, g, data[2], cuda, e, mode='eval', args=args)

    test_(model, loss_func, optimizer, g, data[0], cuda, e, mode='train',args=args)
    test_(model, loss_func, optimizer, g, data[1], cuda, e, mode='eval', args=args)


def train2(model, loss_func, optimizer, g, tr_dataloader, cuda, e):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, 
    min_lr=1e-8, factor=0.001, verbose=True, eps=1e-8)
    
    model.train()

    for i_batch, sample_batched in enumerate(tr_dataloader):
        sample_id, inp_data, out_data, seq_len = sample_batched

        if cuda:
            tensor_x = Variable(torch.FloatTensor(inp_data.float()), requires_grad=True).to(device) 
            tensor_y = Variable(torch.LongTensor(out_data), requires_grad=False).to(device)
        #batch_size = tensor_x.size(0)
        
        # compute model
        logits, _, _, _, _, alpha = model(tensor_x)
        
        ##### EDL Loss Function #####

        #annealing_step = 10
        #global_step = e
        #one_hot_label = one_hot_encoding(tensor_y)
        #loss_batch = loss_func(one_hot_label.to(device), alpha, global_step, annealing_step)
        #loss_batch = torch.mean(loss_batch, dim=0)
        #batch_loss = loss_batch


        ##### Cross Entropy Loss Function #####
        
        #tensor_y_cast = tensor_y.type_as(logits).reshape(-1,1) # for binary cross entropy
         #loss_batch = loss_func(logits, tensor_y.view(-1)) # cross entropy
        
        ##### ELBOLoss Function #####
        loss_batch = nn_ard.ELBOLoss(model, F.cross_entropy)
        loss_batch = loss_batch.forward(logits, tensor_y.view(-1))

        
        # compute accuracy 
        #_, pred_ = logits.max(dim=1)
        #loss += (loss_batch.item() * batch_size)s
        #corr += (pred_.cpu() == tensor_y.view(-1).cpu().data).sum(dim=0).item()
        #total += batch_size
        
        #List_grad.append(model.logits.fully_connected.weight.grad.cpu().numpy())
        #List_weights.append(model.logits.fully_connected.weight.detach().cpu().numpy())

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step() # implement L1 regularizer

        # check if gradient generated
        #model.logits.weight.grad
        
        del loss_batch
        del logits
    print('Sparsification ratio: %.3f%%' % (100.*nn_ard.get_dropped_params_ratio(model)))
    #scheduler.step(metrics=batch_loss, epoch=e)

    #acc = corr / total
    #loss /= total    
    #print('epoch: {}, Train loss:{}, Train acc:{}'
    #.format(e, loss, acc))

def test_(model, loss_func, optimizer, g, data, cuda, e, mode, args, roc_curve=False, 
plot_eer=False, save_scores=True):
    L_logits, L_softmax, L_probs, L_true_labels, L_sigmoid = [], [], [], [], []
    true_negatives, false_positives, false_negatives, true_positives, u, total, loss, acc \
        = 0, 0, 0, 0, 0, 0, 0, 0

    # paths
    logits_score_path = 'scores/{}_logit_scores_{}'.format(mode, e)
    sigmoid_scores_path = 'scores/{}_sigmoid_scores_{}'.format(mode, e)
    softmax_logits_score_path = 'scores/{}_softmax_scores_{}'.format(mode, e)
    probs_logits_score_path = 'scores/{}_probs_scores_{}'.format(mode, e)
    true_score_path = 'scores/{}_labels_{}'.format(mode, e)
    uncertainty_score_path = 'scores/{}_uncertainty_{}'.format(mode, e)

    model.eval()

    for i_batch, sample_batched in enumerate(data):
        sample_id, inp_data, out_data, seq_len = sample_batched

        if cuda:
            tensor_x = Variable(torch.FloatTensor(inp_data.float()), requires_grad=False).to(device)
            tensor_y = Variable(torch.LongTensor(out_data), requires_grad=False).to(device)
        batch_size = tensor_x.size(0)
        
        # model output        
        logits, logsoftmax, softmax, uncertanity, prob, alpha = model(tensor_x)

        # calculate scores
        logits_np     = logits.data.cpu().numpy()
        logsoftmax_np = logsoftmax.data.cpu().numpy() 
        prob_np       = prob.data.cpu().numpy()
        softmax_np    = softmax.data.cpu().numpy()
        scores        = logits[:, 1] - logits[:, 0] # logit scores
        scores_sigmoid = torch.sigmoid(scores).data.cpu().numpy()
        true_labels   = tensor_y.data.cpu().numpy()

        L_logits.append(logits_np)
        L_sigmoid.append(scores_sigmoid)
        L_softmax.append(softmax_np)
        L_probs.append(prob_np)
        L_true_labels.append(true_labels)

        ##### EDL Loss Function #####
        # #annealing_step  = 10*batch_size
        # #global_step = g.get_global_step()

        #annealing_step = 10
        #global_step = e
        #one_hot_label = one_hot_encoding(tensor_y)
        #loss_batch = loss_func(one_hot_label.to(device), alpha, global_step, annealing_step)
        #loss_batch = torch.mean(loss_batch, dim=0)
        #loss += (loss_batch.item() * batch_size)

        ##### Cross Entropy Loss Function #####
        #tensor_y_cast = tensor_y.type_as(logits).reshape(-1,1) # for binary cross entropy 
        #loss_batch = loss_func(logits, tensor_y.view(-1))

        ##### ELBOLoss Function #####
        loss_batch = nn_ard.ELBOLoss(model, F.cross_entropy)
        loss_batch = loss_batch.forward(logits, tensor_y.view(-1))

        loss += loss_batch.item()

        # 2D output
        _, pred_ = logits.max(dim=1)
        acc += (pred_.cpu() == tensor_y.view(-1).cpu().data).sum(dim=0)

        # compute uncertanity
        uncertanity_scalar = np.mean(uncertanity.data.cpu().numpy())
        u += uncertanity_scalar

        # confusion matrix
        pred_numpy = pred_.cpu().numpy() 
        try:
            tn, fp, fn, tp = confusion_matrix(true_labels, pred_numpy).ravel()
        except ValueError:
            pass
        true_negatives  += tn
        false_positives += fp
        false_negatives += fn
        true_positives  += tp 

        total += batch_size

        del loss_batch
        del logits
                    
    t_batch = int(np.round(total / batch_size))
    u      /= t_batch
    acc_    = acc.item() / total
    loss   /= total


    if save_scores:
        np.save(logits_score_path, L_logits)
        np.save(sigmoid_scores_path, L_sigmoid)
        np.save(softmax_logits_score_path, L_softmax)
        np.save(probs_logits_score_path, L_probs)
        np.save(true_score_path, L_true_labels)
        np.save(uncertainty_score_path, u)
         
    print('epoch: {},  {} -> loss:{}, acc:{}, uncer: {}'    
    .format(e, mode, loss, acc_, u))
    print('true_negatives: {}, true_positives: {}, false_negatives: {}, false_positives: {}'
    .format(true_negatives, true_positives, false_negatives, false_positives))
    print('-------')
    print('Sparsification ratio: %.3f%%' % (100.*nn_ard.get_dropped_params_ratio(model)))
    if mode == 'eval': print('--------------------------------------------------------------', os.linesep)

    return {'acc': acc, 'loss': loss}


def model_seed_keeper(worker_id):
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    np.random.seed(worker_id)
    random.seed(worker_id)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(worker_id)
    
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    model_seed_keeper(4)
    
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--bs', '--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--lr', '--learning_rate', default=0.0004, type=float, help='learning rate') # 0.001
    parser.add_argument('--ep', '--epoch', default=110, help='num epoches')
    parser.add_argument('--model', '--train_model', default='LCNN')
    parser.add_argument('--ft', '--feature_extraction', default='MFCC', type=str, help='fetaure extraction')
    parser.add_argument('--ev', '--eval', default=True, type=bool, help='evaluation dataset')
    parser.add_argument('--tr', '--train', default=True, type=bool, help='train dataset')
    parser.add_argument('--ts', '--test', default=True, type=bool, help='test both eval and dev datasets')
    parser.add_argument('--db', '--debug', default=True, type=bool, help='debug the program')
    parser.add_argument('--form', default='np', help='format changer')
    parser.add_argument('--norm', default='mvn', type=str, help='normalizer')
    parser.add_argument('--pd', '--padding', default=64000, type=int, help='padding implementation')
    parser.add_argument('--normalize', default=True, type=bool, help='normalization process ativated')
    
    # generate modes -> NOTE: train mode or test mode
    parser.add_argument('--trModel', type=bool, default=True, help='train datasset mode')
    parser.add_argument('--tsModel', type=bool, default=True, help='test dataset mode')

    # dataset paths, train can be concatenated with development  NOTE: ['train','dev'] 
    parser.add_argument('--tr_x_p', default='train', help='train data path')
    parser.add_argument('--tr_y_p', default='train.trn', help='train data path')

    parser.add_argument('--dev_x_p', default='dev', help='train data path')
    parser.add_argument('--dev_y_p', default='dev.trl', help='train data path')

    parser.add_argument('--eval_x_p', default='eval', help='train data path')
    parser.add_argument('--eval_y_p', default='eval.trl', help='train data path')


    args = parser.parse_args() 

    # args.tr_x_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/ASVspoof2017_V2_wav/train'
    # args.tr_y_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/train.trn.txt'

    # args.dev_x_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/ASVspoof2017_V2_wav/dev'
    # args.dev_y_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/dev.trl.txt'

    # args.eval_x_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/ASVspoof2017_V2_wav/eval'
    # args.eval_y_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/eval.trl.txt'


    # data_paths = LoadDataset(
    #     tr_x=[args.tr_x_p, args.dev_x_p], tr_y=[args.tr_y_p, args.dev_y_p], 
    #     dev_x=[args.dev_x_p], dev_y=[args.dev_y_p], 
    #     eval_x=[args.eval_x_p], eval_y=[args.eval_y_p]
    #     )
    
    # data = ASVDataset(data_paths, args)
    # run_model(data, args)


    # args.tr_x_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/train'
    # args.tr_y_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/train.trn.txt'

    # args.dev_x_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/dev'
    # args.dev_y_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/dev.trl.txt'

    # args.eval_x_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/eval'
    # args.eval_y_p = '/home/audiolab/cagil_asv_journal/asv_spoof_2017_data/cqcc_data/eval.trl.txt'

    data_root = '/home/cagil/Documents/cagil_suslu/Speech_Project/Automated-Speech-Verification_pytorch/cqcc_npy_norm/'

    args.tr_x_p = data_root + '/train'
    args.tr_y_p = data_root + '/train.trn.txt'

    args.dev_x_p = data_root + '/dev'
    args.dev_y_p = data_root + '/dev.trl.txt'

    args.eval_x_p = data_root + '/dev'
    args.eval_y_p = data_root + '/dev.trl.txt'

    batch_size = args.bs
    bin_size = 1  
    n_worker = 0

    tr_dataset = ASVDatasetTorch(args.tr_x_p, args.tr_y_p)
    val_dataset = ASVDatasetTorch(args.dev_x_p, args.dev_y_p)
    eval_dataset = ASVDatasetTorch(args.eval_x_p, args.eval_y_p)

    seq_lengths_tr = tr_dataset.get_seq_lengths()  
    seq_lengths_val = val_dataset.get_seq_lengths() 
    seq_lengths_eval = eval_dataset.get_seq_lengths()        

    sampler_tr = BinnedLengthSampler(seq_lengths_tr, batch_size, batch_size*bin_size)
    sampler_val = BinnedLengthSampler(seq_lengths_val, batch_size, batch_size*bin_size)
    sampler_eval = BinnedLengthSampler(seq_lengths_eval, batch_size, batch_size*bin_size)

    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, 
                        sampler=sampler_tr, collate_fn=asvdataset_collate_fn_pad, 
                        num_workers=n_worker, pin_memory=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
    #                     sampler=sampler_val, collate_fn=asvdataset_collate_fn_pad, 
    #                     num_workers=n_worker, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, 
                        sampler=sampler_eval, collate_fn=asvdataset_collate_fn_pad, 
                        num_workers=n_worker, pin_memory=True)

    #run_model([tr_dataloader, val_dataloader, eval_dataloader], args)
    run_model([tr_dataloader, eval_dataloader], args)
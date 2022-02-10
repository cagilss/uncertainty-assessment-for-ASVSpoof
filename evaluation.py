import numpy as np 
import glob
import re
import os
import matplotlib.pyplot as plt
from sklearn import metrics
import sys 



class Evaluation:

    def __init__(self, p_score, p_label):
        """
        Install bob evrironment: https://www.idiap.ch/software/bob/docs/bob/docs/stable/bob/bob/doc/install.html

        Activate bob environment to use bob.measure.eer
        >>> conda activate bob_py3
        
        Than run evaluation class in bob environment to calculate EER and other evaluation metrics
        This program will provide you a detailed explanation about performance evaluation and graphical
        representation
        
        """
        if not isinstance(p_score, str):
            self.scores = p_score
            self.labels = p_label
        else:
            self.list_s = open(p_score).readlines()
            self.labels = np.load(p_label, allow_pickle=True)
            self.scores = [float(s.strip()) for s in self.list_s]
        
    def EER(self, false_acceptance_rate, thresholds):
        self.intersection(
                FRR=self.false_rejection_rate(thresholds),
                FAR=false_acceptance_rate, 
                thresholds=thresholds
                )

    def false_rejection_rate(self, thresholds):
        genuine = np.count_nonzero(self.labels == 1)
        if genuine == 0:
            genuine = 1
        return [(self.false_rejection(thr)/genuine) 
                for thr in thresholds]

    def false_rejection(self, thr):
        return len([label for score, label in zip(self.scores, self.labels)
                    if score <= thr and label == 1])

    def intersection(self, FRR, FAR, thresholds):
        self.draw_EER(
                FAR=FAR, # False Accept Rate
                FRR=FRR, # False Reject Rate
                thresholds=thresholds, 
                EER=self.intersect(FAR, FRR)
                )

    def intersect(self, FAR, FRR):
        return [far_ for far_, frr_ in zip(FAR, FRR) if far_ >= frr_][0]

    def draw_EER(self, FAR, FRR, thresholds, EER):
        plt.plot(thresholds, FAR)
        plt.plot(thresholds, FRR)
        plt.title('Equal Error Rate: %2.2f %s' % ((EER*100),'%'))
        plt.xlabel('Thresholds')
        plt.ylabel('FAR and FRR')
        plt.show()

    def show_ROC_and_EER(self, typ):
        FPR, TPR, thresholds, score = self.ROC()
        self.EER(FPR, thresholds)
        plt.plot(FPR, TPR)
        plt.title(typ+' '+'roc_ouc_score: %2.4f' % score)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def ROC(self): 
        FPR, TPR, thresholds = metrics.roc_curve(self.labels, self.scores, pos_label=1)
        score = metrics.roc_auc_score(self.labels, self.scores)
        return FPR, TPR, thresholds, score
    
    def plot_roc_curve(self, mode):
        FPR, TPR, thresholds = metrics.roc_curve(self.labels, self.scores, pos_label=1)
        score = metrics.roc_auc_score(self.labels, self.scores)
        plt.plot(FPR, TPR)
        plt.title(mode+' '+'roc_ouc_score: %2.4f' % score)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def show_eer(self):
        return None
    
    def compute_eer(self, pos_label=None):
        FPR, _, thresholds = metrics.roc_curve(self.labels, self.scores, pos_label=pos_label)
        FRR = self.false_rejection_rate(thresholds)
        eer = self.intersect(FPR, FRR)
        return eer
    
    def another_eer(self):
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        from sklearn.metrics import roc_curve
        
        #fpr, tpr, threshold = roc_curve(self.labels, self.scores)

        fpr, tpr, threshold = self.custom_ROC()
        
        auc_score = metrics.auc(fpr, tpr)
        
        fnr = 1 - tpr
        
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        #eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        
        return EER 
     
    def custom_ROC(self):
        thrs = np.arange(0, 1.025, 0.025)
        fpr, tpr = [], []
        for t in thrs:
            count_tp = 0
            count_fp = 0
            for s,l in zip(self.scores,self.labels):
                if s >= t and l == 1: 
                    count_tp += 1
                if s >= t and l == 0:
                    count_fp += 1
            tpr.append(count_tp/self.total_p())  
            fpr.append(count_fp/self.total_p())
        return np.array(fpr), np.array(tpr), thrs

    def total_p(self):
        return len(np.nonzero(self.labels)[0])
    
    def custom_eer(self):
        return 0


    

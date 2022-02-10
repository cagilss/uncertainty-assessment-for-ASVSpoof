import matplotlib.pyplot as plt
import numpy as np
import os
import math 
from sklearn.metrics import confusion_matrix
#import seaborn as sns
    

with open('CE_entropy.txt', 'r') as f:
    x_ce = [float(line.strip()) for line in f.readlines()]

with open('CE_probs.txt', 'r') as f:
    y_ce = [float(line.strip()) for line in f.readlines()]

with open('MSE_entropy.txt', 'r') as f:
    x_mse = [float(line.strip()) for line in f.readlines()]

with open('MSE_probs.txt', 'r') as f:
    y_mse = [float(line.strip()) for line in f.readlines()]



"""
with open('negatives.txt', 'r') as f:
    neg = [float(line.strip()) for line in f.readlines()]

with open('positives.txt', 'r') as f:
    pos = [float(line.strip()) for line in f.readlines()]
"""


"""
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(x_f, y_f, '--')
plt.legend(['MSE'])
plt.xlabel('Empirical CDF of Entropy in ASV 1sth epoch')
plt.subplot(2, 1, 2)
plt.plot(x, y, '--')
plt.legend(['CE'])
plt.xlabel('Empirical CDF of Entropy in ASV 50th epoch')
plt.ylabel('ECDF')
#plt.margins(0.02) 
plt.show()
"""

"""
plt.plot(x, y, '--')
plt.plot(x_f, y_f, '--')
plt.legend(['epoch 1'])
plt.xlabel('Empirical CDF of Entropy in ASV 1sth epoch')

plt.show()
"""

"""
hist, bin_edges = np.histogram(data,bins) # make the histogram

fig,ax = plt.subplots()

# Plot the histogram heights against integers on the x axis
ax.bar(range(len(hist)),hist,width=1) 

# Set the ticks to the middle of the bars
ax.set_xticks([0.5+i for i,j in enumerate(hist)])

# Set the xticklabels to a string that tells us what the bin edges were
ax.set_xticklabels(['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])

plt.show()
"""

"""
plt.plot(x_ce, y_ce)
plt.plot(x_mse, y_mse)
plt.plot(x_edl, y_edl)
plt.legend(['Cross Entropy', 'MSE', 'EDL'])
plt.title('False Positives and False Negatives')
plt.xlabel('Empirical CDF of Entropy')
plt.ylabel('Probability')
#plt.margins(0.02)
plt.show()
"""


#  {‘bar’, ‘barstacked’, ‘step’, ‘stepfilled’}

"""
fig, axis = plt.subplots(1, 2, sharey=True, tight_layout=True)
axis[0].hist(x_ce, histtype='stepfilled', bins=15, density=1, stacked=True, facecolor='g')
axis[0].set_xlabel('Probability')
axis[0].set_ylabel('Misclassified Outputs')
#axis[0].set_title('Probability Distribution Function of the CE Loss')
axis[1].hist(y_ce, histtype='stepfilled', bins=15, density=1, stacked=True, facecolor='g')
axis[1].set_xlabel('Probability')
axis[1].set_ylabel('True Classified Outputs')
#axis[1].set_title('Probability Distribution Function of the CE Loss')

axis[0].grid(True)
axis[1].grid(True)
plt.show()


fig, axis = plt.subplots(1, 2, sharey=True, tight_layout=True)
axis[0].hist(x_mse, histtype='stepfilled', bins=15, density=1, stacked=True, facecolor='g')
axis[0].set_xlabel('Probability')
axis[0].set_ylabel('Misclassified Outputs')
#axis[0].set_title('Probability Distribution Function of the EDL Loss')
axis[1].hist(y_mse, histtype='stepfilled', bins=15, density=1, stacked=True,  facecolor='g')
axis[1].set_xlabel('Probability')
axis[1].set_ylabel('True Classified Outputs')
#axis[1].set_title('Probability Distribution Function of the EDL Loss')

axis[0].grid(True)
axis[1].grid(True)
plt.show()
"""


# true_positive + true_negative vs false_positive + false_negative
plt.plot(x_ce, y_ce, 'b') # true cls
plt.plot(x_mse, y_mse, 'r') # false cls
plt.legend(['True AUC: 0.10', 'False AUC: 0.06'])
# plt.axhline(color='black')
#plt.title('Cumulative Distribution Function of the CE Loss')
plt.xlabel('Empirical CDF of Entropy')
plt.ylabel('Probability')
#plt.margins(0.02)
plt.grid(True)
plt.show()



"""
plt.plot(x_mse, y_mse, 'r')
plt.plot(x_ce, y_ce, 'b')
#plt.title('ROC Curve')
plt.legend(['EDL AUC: 0.80', 'Cross Entropy AUC: 0.66'])
plt.xlabel('False Positive Ratio')
plt.ylabel('True Positive Ratio')
plt.grid(True)
plt.show()
"""


"""
def plot_confusion_matrix(x, y):
    pred = list()
    true_L = np.array([])
    pred_L = np.array([])
    for x_, y_ in zip(x,y):
        pred = np.argmax(x_,axis=1)
        pred_L = np.concatenate((pred_L, pred), axis=0)
        true_L = np.concatenate((true_L, y_), axis=0) 

    cnf_matrix = confusion_matrix(true_L, pred_L) # .ravel()
    group_names = ["True Negative","False Positive","False Negative","True Positive"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cnf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        cnf_matrix.flatten()/np.sum(cnf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cnf_matrix, annot=labels, fmt="", cmap='Blues')

    #plt.title('Confusion Matrix of the Cross-Entropy Loss Predictions')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

"""

if __name__ == "__main__":

    e = 44

    folder = 'best_edl_mse_scores'
    typ = 'eval'
    logit         = '{}_logit_scores_{}.npy'.format(typ, e)
    #sigmoid       = '{}_sigmoid_scores_{}.npy'.format(typ, e)
    #softmax_logit = '{}_softmax_scores_{}.npy'.format(typ, e)
    true_label    = '{}_labels_{}.npy'.format(typ, e)
    
    p1 = os.path.join(folder, logit)
    #p2 = os.path.join(folder, softmax_logit)
    p3 = os.path.join(folder, true_label)
    #p4 = os.path.join(folder, sigmoid)

    x = np.load(p1, allow_pickle=True)
    y = np.load(p3, allow_pickle=True)

    plot_confusion_matrix(x, y)

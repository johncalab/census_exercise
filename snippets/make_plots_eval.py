import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

import joblib
model = joblib.load('income_model.sav')

from utils import load_ttsplit, xyw

train,test = load_ttsplit()
x_train,y_train,w_train = xyw(train)
x_test,y_test,w_test = xyw(test)
y_train_hat = model.predict_proba(x_train)[:,1]
y_test_hat = model.predict_proba(x_test)[:,1]

# make roc_auc, precision_recall, feature importance plots
def make_prec_rec_curve(y, y_hat, fig_size_x=8, fig_size_y=8, save_path=None):
    precision, recall, thresholds = precision_recall_curve(y, y_hat)

    plt.figure(figsize=(fig_size_x,fig_size_y))
    plt.plot(precision, recall, label='precision recall curve')
    plt.xlabel("Precision = TP/(TP+FP) = TP/predicted positive")
    plt.ylabel("Recall = TP/(TP+FN) = TP/actually positive")

    t_half = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(precision[t_half],recall[t_half],'o',label="threshold = 1/2",fillstyle='none', markersize=10)

    plt.legend()
    sns.despine()

    if save_path:
        plt.savefig(save_path)

def make_roc(y_test,y_hat,fig_size_x=8,fig_size_y=8,save_path=None):    
    fpr, tpr, thresholds = roc_curve(y_test,y_hat)

    plt.figure(figsize=(fig_size_x,fig_size_y))
    plt.plot(fpr,tpr,label='ROC curve')
    plt.ylabel('TPR = TP/(TP+FN) = Recall')
    plt.xlabel('FPR = FP/(FP+TN)')

    t_half = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(fpr[t_half],tpr[t_half], 'o', label='threshold = 1/2', fillstyle='none', markersize=10)

    plt.legend()
    sns.despine()
    
    if save_path:
        plt.savefig(save_path)

make_roc(y_test,y_test_hat,save_path='roc.png')
make_prec_rec_curve(y_test,y_test_hat,save_path='prec_rec.png')
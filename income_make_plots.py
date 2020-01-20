print("I will look for:'income_model.sav', 'train.pkl', 'test.pkl', and make roc-auc, prec-rec plots.")

print("Loading data.")
import pandas as pd
train,test = pd.read_pickle('train.pkl'), pd.read_pickle('test.pkl')

from utils import xyw
x_train,y_train,w_train = xyw(train)
x_test,y_test,w_test = xyw(test)

print("Data loaded.")

print("Loading model.")
import joblib
model = joblib.load('income_model.sav')
print("Model loaded.")

print("Computing predictions.")
y_train_hat = model.predict_proba(x_train)[:,1]
y_test_hat = model.predict_proba(x_test)[:,1]
print("Done. Let's move on to generating plots.")


import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

print("Do you wish to also compute permutation feature importances? (this takes a while, enter 'y' for yes)")

def plot_importance(result,cols,fig_x=20,fig_y=8,save_path=None):
    sorted_idx = result.importances_mean.argsort()

    plt.figure(figsize=(fig_x,fig_y))
    plt.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=cols[sorted_idx])

    if save_path:
        plt.savefig(save_path)
        
s = input()
if s == 'y':
    from sklearn.inspection import permutation_importance
    print("Computing result for test set using f1 score. It might take a while.")
    result_test = permutation_importance(model,x_test,y_test,scoring='f1')
    print("Result computed, now producing plot.")
    plot_importance(result_test,cols=x_test.columns,save_path='plots/importance_test.png')
    print("Plot saved. Now computing result for training set, it'll take a while.")

    result_train = permutation_importance(model,x_train,y_train,scoring='f1')
    print("Done, now plotting.")
    plot_importance(result_train,x_train.columns,save_path='plots/importance_train.png')
    print("Plot saved.")


import seaborn as sns
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")

# make roc_auc, precision_recall
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

print("Making roc plot.")
make_roc(y_train,y_train_hat,save_path='plots/roc_train.png')
make_roc(y_test,y_test_hat,save_path='plots/roc_test.png')
print("Done. Saved as 'roc_train.png', 'plots/roc_test.png'.")
print("Making prec-rec plot.")
make_prec_rec_curve(y_train,y_train_hat,save_path='plots/prec_rec_train.png')
make_prec_rec_curve(y_test,y_test_hat,save_path='plots/prec_rec_test.png')
print("Done. Saved as 'prec_rec_train.png', 'prec_rec_test.png'.")

from utils import evaluate_preds
print("Would you like to print out the raw scores? (enter 'y' for yes):")
s = input()
if s == 'y':
    print('Enter a threshold for raw scores (if blank 0.5 will be used):')
    t = input()
    if len(t) == 0:
        t = '0.5'
    t = float(t)

    train_eval = evaluate_preds(y_train_hat,y_train,w_train,t=t)
    test_eval = evaluate_preds(y_test_hat,y_test,w_test,t=t)

    print(f"Computing raw scores, using a threshold of t={t}.")

    with open('plots/scores.txt', 'w') as f:
        s = f"threshold t={t}"
        s += "\ntraining scores:\n"
        s += str(train_eval)
        s += "\ntesting scores:\n"
        s += str(test_eval)
        f.write(s)





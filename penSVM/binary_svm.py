import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import utils

import pickle
import json

COLS = ['E(LUMO-HOMO)_PM6[eV]', 'DipoleMoment [D]', 'IsotropicAverageAlpha [A.U.]', ' AverageGamma [A.U.]',  'INDOS-OscStr(CAS2-1)', 'CI-Coef**2(CAS2-1)', 'T1(CAS2-1) [eV]', 'nH', 'nN', 'nO', 'n5Ring', 'n6Ring', 'nHeteroRing', 'S1-2T1','S1-T1']
TARGET = 'newDRC(PUHF)'

def get_data_train_tst(train_csv, tst_csv):
    
    print('Descriptors: {}'.format(COLS))
    
    train_df = pd.read_csv(train_csv)
    tst_df = pd.read_csv(tst_csv)
    
    train_df["S1-2T1"] = train_df["S1(CAS2-1) [eV]"] - (2*train_df["T1(CAS2-1) [eV]"])
    tst_df["S1-2T1"] = tst_df["S1(CAS2-1) [eV]"] - (2*tst_df["T1(CAS2-1) [eV]"])
    
    train_df["S1-T1"] = train_df["S1(CAS2-1) [eV]"] - train_df["T1(CAS2-1) [eV]"]
    tst_df["S1-T1"] = tst_df["S1(CAS2-1) [eV]"] - tst_df["T1(CAS2-1) [eV]"]
    
    print((train_df["S1-2T1"] < 0).sum())

    X_train = train_df[COLS].to_numpy()
    _Y = train_df[TARGET]
    Y_train = np.array([1 if _>.05 else 0 for _ in _Y])
    
    X_eval = tst_df[COLS].to_numpy()
    _Y = tst_df[TARGET].to_numpy()
    Y_eval = np.array([1 if _>.05 else 0 for _ in _Y])
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    mean, stdev = {}, {}
    for i, d in enumerate(COLS):
        mean[d] = X_train[:, i].mean()
        stdev[d] = X_train[:, i].std()
    out_dict = {}
    out_dict['mean'] = mean
    out_dict['stdev'] = stdev
    with open('MeanStd20220509.json', 'w') as jsonf:
        jsonf.write(json.dumps(out_dict, indent=4))

    X_train = scaler.transform(X_train)
    X_eval = scaler.transform(X_eval)


    return X_train, Y_train, X_eval, Y_eval 


class Metrics():
    
    def __init__(self):
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.total_pos_tst = 0
        self.total_neg_tst = 0
        self.total_pos_pred = 0
        self.total_neg_pred = 0
        self.total_acc = 0.0
        self.model = None

        self.precision = 0.0
        self.recal = 0.0
        self.F1 = 0.0


    def compute_metrics(self, model, X_tst, Y_tst):
        Y_pred = model.predict(X_tst)
        num_true_neg, num_true_pos = 0, 0
        num_false_neg, num_false_pos = 0, 0

        for k in range(Y_tst.shape[0]):
            if Y_tst[k] == 0 and Y_pred[k] == 0:
                num_true_neg += 1
            elif Y_tst[k] == 1 and Y_pred[k] == 1:
                num_true_pos += 1
            elif Y_tst[k] == 1 and Y_pred[k] == 0:
                num_false_neg += 1
            elif Y_tst[k] == 0 and Y_pred[k] == 1:
                num_false_pos += 1
        
        self.model = model

        self.true_pos = num_true_pos
        self.true_neg = num_true_neg
        self.false_pos = num_false_pos
        self.false_neg = num_false_neg
       
        self.total_neg_tst = (Y_tst == 0).sum()
        self.total_pos_tst = (Y_tst == 1).sum()
        self.total_neg_pred = (Y_pred == 0).sum()
        self.total_pos_pred = (Y_pred == 1).sum()
        self.total_acc = metrics.accuracy_score(Y_pred, Y_tst)
        self.precision = metrics.precision_score(Y_pred, Y_tst)
        self.recall = metrics.recall_score(Y_pred, Y_tst)
        self.F1 = metrics.f1_score(Y_pred, Y_tst)


    def __str__(self):
        str_results = 'Model: {}\n'.format(self.model)
        str_results += '\n=======Confusion Matrix=======\n'
        str_results += '{:20s}{:20s}{:20s}\n'.format('','classidied as 1', 'classified as 0')
        str_results += '{:20s}{:15d}{:15d}\n'.format('1 in tst set', self.true_pos, self.false_neg)
        str_results += '{:20s}{:15d}{:15d}\n\n'.format('0 in tst set', self.false_pos, self.true_neg)
        str_results += '{:50s}:{:10d}\n'.format('Total neg in tst set', self.total_neg_tst)
        str_results += '{:50s}:{:10d}\n'.format('Total pos in tst set', self.total_pos_tst)
        str_results += '{:50s}:{:10d}\n'.format('Total neg in pred', self.total_neg_pred)
        str_results += '{:50s}:{:10d}\n\n'.format('Total pos in pred', self.total_pos_pred)
        str_results += '{:50s}:{:10.3f}%\n'.format('True pos / Total pos', 100*(self.true_pos/self.total_pos_tst))
        str_results += '{:50s}:{:10.3f}%\n\n'.format('True neg / Total neg', 100*(self.true_neg/self.total_neg_tst))
        str_results += '{:50s}:{:10.3f}%\n'.format('Total accuracy score',100*self.total_acc)
        str_results += '{:50s}:{:10.3f}\n'.format('Recall = true pos / (true pos + false neg)', self.recall)
        str_results += '{:50s}:{:10.3f}\n'.format('Precision = true pos / (true pos + false pos)', self.precision)
        str_results += '{:50s}:{:10.3f}'.format('F1', self.F1)
        return str_results

if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("usage: {} <training set csv> <test set csv>".format(argv[0]))
        exit(1)

    X_tr, Y_tr, X_tst, Y_tst = get_data_train_tst(sys.argv[1], sys.argv[2])
    
    weights = {0:1.0, 1:4.0}
    clf_lin = svm.SVC( degree=3, gamma=0.20, class_weight=weights) 
    clf_lin.fit(X_tr, Y_tr)
    pickle.dump(clf_lin, open('svm.model', 'wb'))
    m = Metrics()
    m.compute_metrics(clf_lin, X_tst, Y_tst)
    print(m) 

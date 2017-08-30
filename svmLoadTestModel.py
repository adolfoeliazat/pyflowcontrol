import csv, os, sys
import numpy as np
import os
import SVM
import pickle
def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)

filepath = os.path.dirname(os.path.abspath(__file__))

f = open('modelo.model','rb')
model = pickle.load(f)
(data, _) = readData("{}/{}".format(filepath,'i091-2qx2qy.pythonCSVnew.csv'), header=False)
data = data.astype(float)
X, y = data[:,0:-1], data[:,-1].astype(int)
y_hat= model.predict(X)
print(y_hat)

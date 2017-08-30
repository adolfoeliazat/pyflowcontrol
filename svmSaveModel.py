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

csvs = os.listdir(".")
filepath = os.path.dirname(os.path.abspath(__file__))
for csvpath in csvs:
#    wekacsv = open(csv,"r")
    if ".pythonCSV." in csvpath:
        #print(csvpath)
        (data, _) = readData("{}/{}".format(filepath,csvpath), header=False)
        data = data.astype(float)
        X, y = data[:,0:-1], data[:,-1].astype(int)
        model = SVM.SVM(max_iter=10000, kernel_type='linear', C=1.0, epsilon=0.00001)
        #model.fit(X,y)
        f = open('modelo.model', 'wb')
        #pickle.dump(model,f)
        support_vectors, iterations = model.fit(X, y)
        sv_count = support_vectors.shape[0]
        y_hat = model.predict(X)
        pickle.dump(model,f)
        acc = calc_acc(y, y_hat)
        #modelfile = open(csvpath[:4]+".model",'w')
        #modelfile.write(str(model));
        print("File: \t%s\t" % (csvpath), end='')
        print("Support vector count: \t%d\t" % (sv_count), end='')
        print("bias:\t%.3f\t" % (model.b), end='')
        #print("w:\t%s\t" % (str(model.w).rstrip('\n')), end='')
        print("accuracy:\t%.3f\t" % (acc), end='')
        print("Converged after iterations \t%d\t" % (iterations))
'''
print('')
(data, _) = readData("y_haty.txt", header=False)
data = data.astype(float)
X = data[:,0:-1]
y_hat = model.predict(X)
print(y_hat)
(data, _) = readData("y_hatn.txt", header=False)
data = data.astype(float)
X = data[:,0:-1]
y_hat = model.predict(X)
print(y_hat)
'''

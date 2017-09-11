import csv, os, sys
import numpy as np
import os
import SVM
import pickle

epsilon = 0.00001

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

csvs_folder = "samples/diffPythonCSV/"
models_folder = "samples/generatedModels/"
csvs = os.listdir(csvs_folder)
for csvpath in csvs:
    if ".pythonCSV." in csvpath:
        (data, _) = readData("{}/{}".format(csvs_folder,csvpath), header=False)
        data = data.astype(float)
        X, y = data[:,0:-1], data[:,-1].astype(int)
        model = SVM.SVM(max_iter=10000, kernel_type='linear', C=1.0, epsilon=epsilon)
        model.fit(X,y)
        epsilon_base_name = 'e'+str(epsilon)+".model"
        f = open(models_folder+csvpath.replace(".pythonCSV.csv",'-'+epsilon_base_name), 'wb')
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

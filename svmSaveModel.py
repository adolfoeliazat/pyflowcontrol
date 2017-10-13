import csv, os, sys
import numpy as np
import os
import SVM
import pickle
import patternrecognition as pr
import random
import time

def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))
#'''
def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)
'''
def calc_acc(y, y_hat):
    acertos = 0
    for i in range(len(y)):
        if y[i]==y_hat[i]:
            acertos+=1
    return float(acertos)/float(len(y))
#'''

def frange(start,stop,step):
    rangelist = []
    i = start
    while i < stop:
        rangelist.append(i)
        i = i * step
    return rangelist
        
def gridSearch(conjunto_treino,gabarito_treino,conjunto_teste,gabarito_teste,C_min,C_max,epsilon_min,epsilon_max):
    
    epsilon_list = frange(epsilon_min,epsilon_max,epsilon_step)
    C_list = []
    for i in range(C_min,C_max+C_step,C_step):
        C_list.append(i)
    while C_min < C_max:
        #Calculando acuracia pro valor minimo de C e epsilon
        model = SVM.SVM(max_iter=10000, kernel_type='linear', C=C_min, epsilon=epsilon_max)
        support_vectors, iterations = model.fit(conjunto_treino,gabarito_treino)
        sv_count = support_vectors.shape[0]
        y_hat = model.predict(conjunto_teste)
        acc_Cmin = calc_acc(gabarito_teste, y_hat)

        #Calculando acuracia pro valor maximo de C e epsilon
        model = SVM.SVM(max_iter=10000, kernel_type='linear', C=C_max, epsilon=epsilon_max)
        support_vectors, iterations = model.fit(conjunto_treino,gabarito_treino)
        sv_count = support_vectors.shape[0]
        y_hat = model.predict(conjunto_teste)
        acc_Cmax = calc_acc(gabarito_teste, y_hat)

        if acc_Cmin < acc_Cmax:
            C_list = C_list[int(len(C_list)/2):]
            C_min = C_list[0]
        else:
            C_list = C_list[:int(len(C_list)/2)]
            C_max = C_list[len(C_list)-1]
    acc = 0
    while epsilon_min < epsilon_max:
        model = SVM.SVM(max_iter=10000, kernel_type='linear', C=C_min, epsilon=epsilon_min)
        support_vectors, iterations = model.fit(conjunto_treino,gabarito_treino)
        sv_count = support_vectors.shape[0]
        y_hat = model.predict(conjunto_teste)
        acc_emin = calc_acc(gabarito_teste, y_hat)

        #Calculando acuracia pro valor maximo de C e epsilon
        model = SVM.SVM(max_iter=10000, kernel_type='linear', C=C_min, epsilon=epsilon_max)
        support_vectors, iterations = model.fit(conjunto_treino,gabarito_treino)
        sv_count = support_vectors.shape[0]
        y_hat = model.predict(conjunto_teste)
        acc_emax = calc_acc(gabarito_teste, y_hat)

        if acc_emin < acc_emax:
            epsilon_list = epsilon_list[int(len(C_list)/2):]
            epsilon_min = epsilon_list[0]
            acc = acc_emax
        else:
            epsilon_list = epsilon_list[:int(len(epsilon_list)/2)]
            epsilon_max = epsilon_list[len(epsilon_list)-1]
            acc = acc_emin
    
    print("File: \t%s\t" % (lbps_path), end='')
    print("Support vector count: \t%d\t" % (sv_count), end='')
    print("bias:\t%.3f\t" % (model.b), end='')
    #print("w:\t%s\t" % (str(model.w).rstrip('\n')), end='')
    print("accuracy:\t%.3f\t" % (acc), end='')
    print("C: ",C_min," eps: ",epsilon_min)
    #print("Converged after iterations \t%d\t" % (iterations))
    #epsilon_base_name = 'e'+str(epsilon)+".model"
    #f = open(models_folder+lbps_path.replace(".pythonCSV.csv",'-'+epsilon_base_name), 'wb')
    #pickle.dump(model,f)
    return C_min,epsilon_min,acc

amostras_positivas_treino = 5
amostras_positivas_teste = 5
epsilon_min = 0.0000001
epsilon_max = 0.00001
epsilon_step = 10
C_min = 1
C_max = 10
C_step = 1
acc_avg = 0
accuracies = []
Cs = []
eps = []

lbps_folder = "samples/LBPind/"
models_folder = "samples/generatedModels/"
lbps_src = os.listdir(lbps_folder)
lbps_src.sort()

start_time = time.perf_counter()
for lbps_path in lbps_src:
    if ".lbp" in lbps_path:
        diff_lbps_pos = []
        diff_lbps_pos_extra = []
        diff_lbps_neg = []
        diff_lbps_test = []
        flbps = open(lbps_folder+lbps_path,"rb")
        #print("\n==================================================")
        #print('\tIndividuo: '+lbps_folder+lbps_path)
        #print("==================================================")
        lbps = pickle.load(flbps)
        #print('pos size:'+str(len(lbps)))
        for lbp in lbps:
            if lbp is not None:
                diferenca = pr.diferenca(lbp,lbps[0])
                #print('diferenca'+str(diferenca))
                if len(diff_lbps_pos)<amostras_positivas_treino:
                    diff_lbps_pos.append(diferenca) #calcula a diferenca absoluta de cada histograma com o da foto referencia
                else:
                    #carregar o vetor positivos pra testes
                    diff_lbps_pos_extra.append(diferenca)
        flbps.close()
        diff_lbps_pos_extra = diff_lbps_pos_extra[:amostras_positivas_teste]
        diff_lbps_test.extend(diff_lbps_pos_extra)
        #print("\tAMOSTRAS NEGATIVAS")
        #print("==================================================")
        #calculando diferencas do individuo negativo sorteado aleatoriamente
        while len(diff_lbps_pos) != len(diff_lbps_neg): #for i in range(len(diff_lbps_pos)):
            amostra_aleatoria = random.randint(0,len(lbps_src)-1)
            if lbps_path != lbps_src[amostra_aleatoria]:
                #print(lbps_folder+lbps_src[amostra_aleatoria],end='')
                flbps_neg = open(lbps_folder+lbps_src[amostra_aleatoria],'rb')
                lbps_neg = pickle.load(flbps_neg)
                #print('negsize '+str(len(diff_lbps_neg)))

                #adicionando amostras negativas ao treino
                amostra_aleatoria_negativa = random.randint(0,len(lbps_neg)-1)
                while lbps_neg[amostra_aleatoria_negativa] is None:
                    amostra_aleatoria_negativa = random.randint(0,len(lbps_neg)-1)
                #print(' '+str(amostra_aleatoria_negativa),end='')
                diff_lbps_neg.append(pr.diferenca(lbps_neg[amostra_aleatoria_negativa],lbps[0]))

                #adicionando as amostras negativas ao teste
                amostra_teste = random.randint(0,len(lbps_neg)-1)
                while lbps_neg[amostra_teste] is None:
                    amostra_teste = random.randint(0,len(lbps_neg)-1)
                if len(diff_lbps_test)-len(diff_lbps_pos_extra)<len(diff_lbps_pos_extra):
                    #print(' '+str(amostra_teste))
                    diff_lbps_test.append(pr.diferenca(lbps_neg[amostra_teste],lbps[0]))
                flbps_neg.close()
        #for i in range(2*len(diff_lbps_post)):
        todas_amostras =[]
        todas_amostras.extend(diff_lbps_pos)
        todas_amostras.extend(diff_lbps_neg)
        
        conjunto_treino = (np.array(todas_amostras))#.astype(int)
        gabarito_treino = np.array([1]*len(diff_lbps_pos)+[-1]*len(diff_lbps_neg))
        conjunto_teste = np.array(diff_lbps_test)
        gabarito_teste = np.array([1]*len(diff_lbps_pos_extra)+[-1]*(len(diff_lbps_test)-len(diff_lbps_pos_extra)))
        C,epsilon,acc = gridSearch(conjunto_treino,gabarito_treino,conjunto_teste,gabarito_teste,C_min,C_max,epsilon_min,epsilon_max)
        accuracies.append(acc)
        Cs.append(C)
        eps.append(epsilon)

        '''
        print('lbplenpos: '+str(len(diff_lbps_pos)))
        print('lbplenneg: '+str(len(diff_lbps_neg)))
        print(todas_amostras)
        print(len(todas_amostras))
        print(classes)
        print(len(classes))
        '''
        
        
end_time = time.perf_counter()
total_time = end_time-start_time
acc_avg = sum(accuracies)/len(accuracies)
C_avg = sum(Cs)/len(Cs)
eps_avg = sum(eps)/len(eps)
print("accuracy average",acc_avg)
print("Cs average",C_avg)
print("epsilon average",eps_avg)
print("total time elapsed: "+str(total_time))

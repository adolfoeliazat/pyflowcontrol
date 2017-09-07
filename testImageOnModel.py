import os
import patternrecognition as pr
import pickle
import SVM
import cv2
import numpy as np

path = "samples/testImages/i091/"
model = pickle.load(open("samples/diffPythonCSV/i092-2qx2qy.model",'rb'))
faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
quadrantesX = 2
quadrantesY = 2
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

def lbp(file):
	print(file)
	image = cv2.imread(file)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ## EQUALIZA IMAGEM
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	gray = clahe.apply(gray)
	#gray = cv2.equalizeHist(gray)
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	)
	if(len(faces)==0):
            #x = [1000]*num_caracteristicas
            return None
	else:
	    for (x, y, w, h) in faces:
	        aux=gray[y:y+h,x:x+w]
	    #print(aux)
	    cv2.imshow("",aux)
	    cv2.waitKey(0)
	    vetor=pr.calculaLBP(aux,quadrantesX,quadrantesY)
	    return vetor

	

files = os.listdir(path);
difs = [None]*(len(files)-2)
for file in files:
    if "qa" in file:
        imgref=lbp(path+file)
i=0
for file in files:
    if ".jpg" in file and "qa" not in file:
        currlbp = lbp(path+file)
        if(currlbp) is not None:
            difs[i]=pr.diferenca(currlbp,imgref)
            i=i+1
        else:
            difs[i] = [255.0]*1024
            i=i+1
print(np.array(difs))
print(type(np.array(difs)))
y_hat = model.predict(np.array(difs))
print(y_hat)
        

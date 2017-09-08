import os
import patternrecognition as pr
import pickle
import SVM
import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
quadrantesX = 2
quadrantesY = 2
modelos = []
success_rates = []
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
	    #cv2.imshow("",aux)
	    #cv2.waitKey(0)
	    vetor=pr.calculaLBP(aux,quadrantesX,quadrantesY)
	    return vetor

	

def test_model(testModel,testImages):
    #model_to_test = str(input("Insert the model number you are trying to test (ex: i080): "))
    #images_to_test = str(input("Insert the images number you are using to test (ex: i080): "))
    model_to_test = '{:03d}'.format(testModel)
    images_to_test = '{:03d}'.format(testImages)
    path = "samples/muct/i"+images_to_test+"/"
    model_path = "samples/generatedModels/i"+model_to_test+"-2qx2qy.model"
    modelos.append(model_path)
    model = pickle.load(open(model_path,'rb'))

    files = os.listdir(path);
    #print(files)

    for file in files:
        if "qa" in file:
            imgref=lbp(path+file)
            #print('achou')
    difs = []
    for file in files:
        if ".jpg" in file and "qa" not in file:
            currlbp = lbp(path+file)
            if(currlbp) is not None:
                difs.append(pr.diferenca(currlbp,imgref))
            else:
                difs.append([255.0]*quadrantesX*quadrantesY*256)
    #print(np.array(difs))
    #print(type(np.array(difs)))
    y_hat = model.predict(np.array(difs))
    successes = 0
    for number in y_hat:
        if number == 1:
            successes = successes +1
    print(successes / len(y_hat))
    success_rates.append(successes / len(y_hat))
    print(y_hat)

for i in range(91):
    test_model(i,i)
print("=================================")
print("\tRESULTADO FINAL")
print("=================================")
for i in range(len(modelos)):
    print(modelos[i]+"\t"+str(success_rates[i]))
    

import os
import patternrecognition as pr
import pickle
import SVM
import cv2
import numpy as np

# Here we define if you are testing the model using it's own images (randomic = False) or other images (randomic = True)
randomic = True
# Here you can define the inferior and superior individual sample number (ie: 0 and 90 for i000 through i090)
sample_inferior = 0
sample_superior = 90
faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
quadrantesX = 2
quadrantesY = 2
desejado_positivas = 15
desejado_negativas = 30
epsilon = 0.00001

q_base_name = str(quadrantesX)+'qx'+str(quadrantesY)+'qy'
PN_base_name = str(desejado_positivas)+'P'+str(desejado_negativas)+'N'
epsilon_base_name = 'e'+str(epsilon)
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

	

def test_model(testModel,testImages): # this function receives model number and which set of images it is used to test
    model_to_test = '{:03d}'.format(testModel)
    images_to_test = '{:03d}'.format(testImages)
    path = "samples/muct/i"+images_to_test+"/"
    model_path = "samples/generatedModels/i"+model_to_test+"-"+q_base_name+"-"+PN_base_name+"-"+epsilon_base_name+".model"
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
    true_success = 0.0
    if randomic == True:
        true_success = 1.0-(successes / len(y_hat))
    else:
        true_success = (successes / len(y_hat))

    print(true_success)
    success_rates.append(true_success)
    print(y_hat)


if randomic == True:
    for i in range(sample_inferior,sample_superior+1):
        if i == sample_superior:
            test_model(i,sample_inferior)
        else:
            test_model(i,i+1)
else:
    for i in range(sample_inferior,sample_superior+1): # loop through models i000 and i090
        test_model(i,i)
print("=================================")
print("\tFINAL RESULT")
print("=================================")
for i in range(len(modelos)): # prints out the final results
    print(modelos[i]+"\t"+str(success_rates[i]))
    

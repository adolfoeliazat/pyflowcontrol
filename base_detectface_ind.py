#!/usr/bin/env python3
import cv2
import patternrecognition as pr
import os


def openBase(base_path):
    try:
        base_file = open(base_path,"r")
    except FileNotFoundError as exception:
        base_file = open(base_path,"a")
        header_string = ""
        for i in range(0,1024):
            header_string += ", var"+str(i)                       
        base_file.write("name " + header_string + "\n")
    base_file.close()
    base_file = open(base_path,"a")
    return base_file

def lbp(file):
	#print(file)
	image = cv2.imread(file)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	)
	if(len(faces)==0):
		return None 
	else:
		for (x, y, w, h) in faces:
		    aux=gray[y:y+h,x:x+w]
		print(aux)

		#cv2.imshow("",aux)
		#cv2.waitKey(0)
		vetor=pr.calculaLBP(aux,1,1)
		return vetor

	

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
folderpath = 'bi/base_dividida/'
dest='bi/jpg-faces-grey'
ind_folder = os.listdir(folderpath)
histograms = [None]*15
ids = [None]*15
reference_index=0;
for ind in ind_folder: # percorre a pasta que contem as pastas com fotos de cada individuo
	print('pasta '+ind)
	pictures = os.listdir(folderpath+ind)
	i=0
	for x in pictures: # percorre a pasta com as fotos do individuo
		print(x)
		if(x[5]=='c' and x[6]=='s' and x[7]=='v'): # verifica se o arquivo nao é csv
			print('base')
			i=i-1
		else:
			if(x[4]=='q' and x[5]=='a'): # verifica se a foto é a foto referencia
				reference_index=i # salva indice da foto referencia
				print('reference_index: ',reference_index)			
			histograms[i]=lbp(folderpath+'/'+ind+'/'+x) # calcula histograma lbp de cada imagem 
			ids[i]=x;
		i=i+1
		
	j=0;
	base_ind=openBase(folderpath+ind+'/'+ind+'.csv')
	for h in histograms: # percorre os histogrmas
		dif=pr.diferenca(h,histograms[reference_index]) #calcula a diferenca absoluta de cada histograma com o da foto referencia 
		base_ind.write(ids[j]+','+','.join(map(str,dif))+"\n") # grava na base
		j=j+1
	reference_index=0;
	histograms = [None]*15
	ids = [None]*15
	
		
		
	


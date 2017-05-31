#!/usr/bin/env python3
import cv2
import patternrecognition as pr
import os
import random

imagens_positivas = 15
imagens_negativas = 30
quadrantesX=2
quadrantesY=2
base_name = str(quadrantesX)+"qx"+ str(quadrantesY)+"qy"
num_caracteristicas=quadrantesX*quadrantesY*256
def createBase(base_path):
    base_file = open(base_path,"w")
    header_string = ""
    for i in range(0,num_caracteristicas):
        header_string += ", var"+str(i)                       
    base_file.write("name " + header_string + "\n")
    base_file.close()
    base_file = open(base_path,"a")
    return base_file

def openBase(base_path):
    try:
        base_file = open(base_path,"r")
    except FileNotFoundError as exception:
        base_file = open(base_path,"w")
        header_string = ""
        for i in range(0,num_caracteristicas):
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

	

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
folderpath = '../base_dividida/'
dest='./jpg-faces-grey/'
dir_aleatorio='../muct/'
ind_folder = os.listdir(folderpath)
histograms = [None]*(imagens_positivas+imagens_negativas)
ids = [None]*(imagens_positivas+imagens_negativas)
reference_index=0;
for ind in ind_folder: # percorre a pasta que contem as pastas com fotos de cada individuo
	print('pasta '+ind)
	pictures = os.listdir(folderpath+ind)
	aleatorias = os.listdir(dir_aleatorio)
	#print(pictures)
	for x in range(0,len(pictures)):
		pictures[x]=folderpath+ind+'/'+pictures[x]	
	for x in range(0,imagens_negativas):
		aux= random.randint(0,len(aleatorias)-1)
		pictures.append(dir_aleatorio+aleatorias[aux])		
	i=0
	#print(pictures)
	for x in pictures: # percorre a pasta com as fotos do individuo
		print(x)
		if(x[-3:]=="csv"): # verifica se o arquivo nao é csv
			print('base')
			i=i-1
		else:
			#print(x[-9:-7])		
			if(x[-9:-7]=="qa"): # verifica se a foto é a foto referencia
				reference_index=i # salva indice da foto referencia
				print('reference_index: ',reference_index)			
			#print(i)
			histograms[i]=lbp(x) # calcula histograma lbp de cada imagem 
			ids[i]=x;
		i=i+1
		
	j=0;
	base_ind=createBase(folderpath+ind+'/'+ind+'-'+base_name+'.csv')
	for h in histograms: # percorre os histogrmas
		if h is None:
			pass
		else:
			dif=pr.diferenca(h,histograms[reference_index]) #calcula a diferenca absoluta de cada histograma com o da foto referencia 
			#print("ids" + ids[j][-13:-9])
			#print("ind" + ind)
			if ids[j][-13:-9] == ind:
				base_ind.write('y'+','+','.join(map(str,dif))+"\n") # grava na base
			else:
				base_ind.write('n'+','+','.join(map(str,dif))+"\n") # grava na base
		j=j+1
	reference_index=0;
	histograms = [None]*(imagens_positivas+imagens_negativas)
	ids = [None]*(imagens_positivas+imagens_negativas)
	
 
		
		
	


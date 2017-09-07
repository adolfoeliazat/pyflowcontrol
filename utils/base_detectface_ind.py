#!/usr/bin/env python3
import cv2
import patternrecognition as pr
import os
import random

# especificacao de quantas imagens positivas e quantas imagens negativas se quer extrair os dados.
# Caso a pasta relativa a imagem contenha menos imagens do que o desejado, a quantidade sera
# reduzida ao numero de imagens existentes
desejado_positivas = 15
desejado_negativas = 45

imagens_positivas = desejado_positivas
imagens_negativas = desejado_negativas
quadrantesX=2
quadrantesY=2
base_name = str(quadrantesX)+"qx"+ str(quadrantesY)+"qy"
num_caracteristicas=quadrantesX*quadrantesY*256
faceCascade = cv2.CascadeClassifier('../cascades/haarcascade_frontalface_default.xml')
folderpath = '../samples/muct/'
dir_aleatorio='../samples/muct/'
output_csv_folder='../samples/diffPythonCSV/'
ind_folder = os.listdir(folderpath)
histograms = [None]*(imagens_positivas+imagens_negativas)
ids = [None]*(imagens_positivas+imagens_negativas)

def createBase(base_path):
    base_file = open(base_path,"w")
    if "pythonCSV" not in base_path:
        header_string = ""
        for i in range(0,num_caracteristicas):
            header_string += "var"+str(i)+", "
        base_file.write(header_string + " name\n")

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
            header_string += "var"+str(i)+ ", "
        base_file.write(header_string + " name\n")
    base_file.close()
    base_file = open(base_path,"a")
    return base_file

def lbp(file):
    #print(file)
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ## EQUALIZA IMAGEM
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    faces = faceCascade.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5,)
    if(len(faces)==0):
            #x = [1000]*num_caracteristicas
            print(" <=== FACE NAO DETECTADA, REMOVIDO DA BASE",end='')
            return None
    else:
        for (x, y, w, h) in faces:
            aux=gray[y:y+h,x:x+w]
        #print(aux)
        #cv2.imshow("",aux)
        #cv2.waitKey(0)
        vetor=pr.calculaLBP(aux,quadrantesX,quadrantesY)
        return vetor
   

##############################
###     HERE IT STARTS    ####
##############################

for ind in ind_folder: # percorre a pasta que contem as pastas com fotos de cada individuo
    print("==============================")
    print("\tIMAGEM:\t"+ind)
    print("==============================")
    print("\t  POSITIVAS")
    print("==============================")
    imagens_positivas = desejado_positivas
    imagens_negativas = desejado_negativas
    pictures = os.listdir(folderpath+ind)
    randir = os.listdir(dir_aleatorio)
    #print(pictures)
    for picture in pictures: #remove arquivos nao .jpg da lista de imagens
        if ".jpg" not in picture:
            #print("not jpg: "+picture)
            pictures.remove(picture)
    for picture in pictures: #redefine o path certo
        pictures[pictures.index(picture)]=folderpath+ind+'/'+picture
    num_positivas = len(pictures)
    if num_positivas < imagens_positivas:
        imagens_positivas = num_positivas
    #print("IMG POSITIVAS:"+str(imagens_positivas))
    for x in range(0,imagens_negativas): # Esse loop faz acesso aleatorio na pasta de samples e confere se é .jpg e não é relativo ao indivíduo
        aux= random.randint(0,len(randir)-1)
        randir2 = os.listdir(dir_aleatorio+randir[aux])
        aux2= random.randint(0,len(randir2)-1)
        picpath = dir_aleatorio+randir[aux]+"/"+randir2[aux2]
        while ".jpg" not in picpath or ind in picpath:
            aux= random.randint(0,len(randir)-1)
            randir2 = os.listdir(dir_aleatorio+randir[aux])
            aux2= random.randint(0,len(randir2)-1)
            picpath = dir_aleatorio+randir[aux]+"/"+randir2[aux2]
        pictures.append(picpath)
    #print(pictures)
    i=0
    reference_index=0;
    histograms = []
    #[None]*(imagens_positivas+imagens_negativas)
    ids = []
    #ids = [None]*(imagens_positivas+imagens_negativas)
    for x in pictures: # abre as imagens positivas e negativas, acha a referencia e calcula o lbp pra todas
        print(x,end=' ')
        print(" "+str(i),end='')
        histograms.append(lbp(x)) # calcula histograma lbp de cada imagem 
        if "qa" in x and ind in x: # verifica se a foto é a foto referencia e pertence ao indivíduo
            reference_index = i # salva indice da foto referencia
            print('<== IMAGEM REFERENCIA: ',reference_index,end='')            
        if i == imagens_positivas-1:
            print("\n==============================")
            print("\t  NEGATIVAS")
            print("==============================",end='')
        
        ids.append(x);
        i=i+1
        print('')
        
    j=0;
    #base_ind=createBase(folderpath+ind+'/'+ind+'-'+base_name+'.csv')
    #base_python=createBase(folderpath+ind+'/'+ind+'-'+base_name+'.pythonCSV.csv')
    base_ind=createBase(output_csv_folder+ind+'-'+base_name+'.csv')
    base_python=createBase(output_csv_folder+ind+'-'+base_name+'.pythonCSV.csv')
    for h in histograms: # percorre os histogrmas
        if h is None:
            print("passou nulo")
            pass
        else:
            dif=pr.diferenca(h,histograms[reference_index]) #calcula a diferenca absoluta de cada histograma com o da foto referencia 
            #print("ids" + ids[j][-13:-9])
            #print("ind" + ind)
            if ind in ids[j]:
                string = str(dif).replace('[','').replace(']','')#.replace('.0','')
                base_ind.write(string +", y\n") # grava na base
                base_python.write(string +", 1\n")
            else:
                string = str(dif).replace('[','').replace(']','')#.replace('.0','')
                base_ind.write(string+", n\n") # grava na base
                base_python.write(string+", -1\n")
        j=j+1
    
    
 
        
        
    






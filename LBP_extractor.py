#Simple python Script
#!/usr/bin/env python3
import cv2
import patternrecognition as pr
import os
import random
import pickle

# especificacao de quantas imagens positivas e quantas imagens negativas se quer extrair os dados.
# Caso a pasta relativa a imagem contenha menos imagens do que o desejado, a quantidade sera
# reduzida ao numero de imagens existentes
desejado_positivas = 15
desejado_negativas = 0

imagens_positivas = desejado_positivas
imagens_negativas = desejado_negativas
quadrantesX=2
quadrantesY=2
base_name = str(quadrantesX)+"qx"+ str(quadrantesY)+"qy"
num_caracteristicas=quadrantesX*quadrantesY*256
faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
folderpath = 'samples/rlSamples/'
dir_aleatorio='samples/muct/'
output_python_csv_folder='samples/diffPythonCSV/'
output_weka_csv_folder='samples/diffWekaCSV/'
output_lbp_folder='samples/LBPtest/'
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
        maiorarea=0
        xaux=0
        yaux=0
        waux=0
        haux=0
        for (x, y, w, h) in faces:
            if w*h > maiorarea:
                maiorarea=w*h
                xaux=x
                yaux=y
                waux=w
                haux=h
        aux=gray[yaux:yaux+haux,xaux:xaux+waux]
        #print(aux)
        #cv2.imshow("",aux)
        #cv2.imwrite(file.replace(".jpg",'')+'-face.jpg',aux)        #cv2.waitKey(0)
        vetor=pr.calculaLBP(aux,quadrantesX,quadrantesY)
        #print(vetor)
        return vetor
   

##############################
###     HERE IT STARTS    ####
##############################
ind_folder.sort()
for ind in ind_folder: # percorre a pasta que contem as pastas com fotos de cada individuo
    print("\n==================================================")
    print("\tIMAGEM:\t"+ind)
    #print("==============================")
    print("\t  AMOSTRAS")
    print("==================================================")
    imagens_positivas = desejado_positivas
    imagens_negativas = desejado_negativas
    pictures = os.listdir(folderpath+ind)
    #print(pictures)
    randir = os.listdir(dir_aleatorio)
    #print(pictures)
    for picture in pictures: #remove arquivos nao .jpg da lista de imagens
        if ".jpg" not in picture and ".JPG" not in picture:
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
        #print(" "+str(i),end='')
        histograms.append(lbp(x)) # calcula histograma lbp de cada imagem
        
        if "qa" in x and ind in x: # verifica se a foto é a foto referencia e pertence ao indivíduo
            reference_index = i # salva indice da foto referencia
            print(' <== IMAGEM REFERENCIA',end='')            
        ids.append(x);
        i=i+1
        print('')
        
    #faz swap do histograma da imagem de referencia para o primeiro histograma
    # e entao salva os histogramas em um arquivo binario
    strcfg = base_name+'-'+str(desejado_positivas)+'P'+str(desejado_negativas)+'N'
    aux=histograms[0]
    histograms[0]=histograms[reference_index]
    histograms[reference_index]=aux
    f = open(output_lbp_folder+ind+"-"+base_name+".lbp","wb")
    pickle.dump(histograms,f)
    reference_index=0
    print('\n-> Arquivo LBP: '+output_lbp_folder+ind+"-"+base_name+".lbp")
    #print("__________________________________________________")
print("==================================================")
    
    

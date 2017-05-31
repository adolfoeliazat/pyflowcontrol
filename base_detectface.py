#!/usr/bin/env python3
import cv2
import time
import patternrecognition as pr
import os

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
folderpath = '../base_dividida/i016/'
dest='../jpg-faces-grey/'
filename = os.listdir(folderpath)
for x in filename:
	print(x)

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

base_file = openBase("base-teste.csv")

for x in filename:
	person_name=x[:4]
	print(person_name)
	image = cv2.imread(folderpath+x)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	)

	for (x, y, w, h) in faces:
		aux=gray[y:y+h,x:x+w]

	
	cv2.imwrite(dest+"/"+str(x)+".jpg",aux)

	vetor=pr.calculaLBP(aux,2,2)
	base_file.write(person_name+','+','.join(map(str,vetor))+"\n")
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

	cv2.imshow('Faces found' ,aux)
	cv2.waitKey(0)

base_file.write("\n")
base_file.close()

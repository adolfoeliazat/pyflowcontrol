#!/usr/bin/env python3
import io
import cv2
import numpy
import time
import patternrecognition as pr
import sys

# Program adjustable variables
grayscales = 1.5
totalcameras = 1
warming_time = 2
file = open("base.csv","a")

# Program definitions
rosto = []
cap = []
counter = 0
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Opening windows and warming up cams
for i in range(totalcameras):
    cap.append(cv2.VideoCapture(i))
    cv2.namedWindow("Cam "+str(i))
warming_end = time.time()+warming_time
while warming_end > time.time():
    for i in range(totalcameras):
        ret,frame = cap[i].read()
        rows,cols,channels = frame.shape
        cv2.putText(frame,"WARMING UP", (int(cols/2-200),int(rows/2-10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255))
        cv2.imshow("Cam "+str(i),frame)

# Main Loop
print("Hit 'q' to quit...\n")
nome = input("Digite o nome da pessoa que sera escaneada: ")
while True:
    for i in range(totalcameras):
        ret,frame = cap[i].read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, grayscales, 5)
        #print("Found "+str(len(faces))+" face(s)")
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            npaux=gray[y:y+h,x:x+w]
            #rostoaux=cv2.cv.fromarray(gray[x:y,x+w:y+h])
            rosto.append(npaux)
            if counter==0:
                print(nome + " detected")
                vetor=pr.calculaLBP(rosto[0],2,2)
                cv2.putText(frame,nome, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            else:
                vetor2=pr.calculaLBP(rosto[counter],2,2)
                distancia=pr.distanciaEuclidiana(vetor,vetor2)
                if distancia<0.1:
                    cv2.putText(frame,nome, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                    file.write(nome+","+','.join(map(str,vetor2))+"\n")
                else:
                    cv2.putText(frame,"Outra Pessoa", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                print("Distancia Euclidiana: "+str(distancia))
            cv2.imshow("Cam "+str(i),frame)
            counter+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#file.write(nome+","+','.join(map(str,vetor2))+"\n\n") 

for i in range(totalcameras):
    cap[i].release()
    cv2.destroyAllWindows()

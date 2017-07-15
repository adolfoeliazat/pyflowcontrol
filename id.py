#!/usr/bin/env python3
import io
import cv2
import numpy
import time
import patternrecognition as pr
import sys
import os

# Program adjustable variables
grayscales = 2
totalcameras = 2
warming_time = 2
base_path = "base.csv"
qx = 2
qy = 2
caracteristicas=256*qx*qy

# Program definitions
rosto = []
cap = []
counter = 0
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

# Opening windows and warming up cams
def openNwarm():
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

def openBase(base_path):
    try:
        base_file = open(base_path,"r")
    except FileNotFoundError as exception:
        base_file = open(base_path,"a")
        header_string = ""
        for i in range(0,caracteristicas):
            header_string += ", var"+str(i)                       
        base_file.write("name " + header_string + "\n")
    base_file.close()
    base_file = open(base_path,"a")
    return base_file
    
# Make sure path exists, if it doesnt, then create it
def createPath(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        print("Person path found.\n")

# Main Loop
def Main():
    openNwarm()
    print("Hit 'q' to quit...\n")
    person_name = input("Digite o nome da pessoa que sera escaneada: ")
    person_path = "sample_images/" + str(person_name) + "/"
    createPath(person_path)
    base_file = openBase(base_path)
    counter = 0
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
                    print(person_name + " detected")
                    vetor=pr.calculaLBP(rosto[0],2,2)
                    cv2.putText(frame,person_name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                else:
                    vetor2=pr.calculaLBP(rosto[counter],2,2)
                    distancia=pr.distanciaEuclidiana(vetor,vetor2)
                    if distancia<0.1:
                        cv2.putText(frame,person_name, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                        imgstring=time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())+".jpg"
                        cv2.imwrite(person_path+imgstring,npaux)
                        base_file.write(person_name+","+','.join(map(str,vetor2))+"\n")
                    else:
                        cv2.putText(frame,"Outra Pessoa", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                    print("Distancia Euclidiana: "+str(distancia))
                cv2.imshow("Cam "+str(i),frame)
                counter+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            base_file.write("\n")
            base_file.close()
            break
        
    for i in range(totalcameras):
        cap[i].release()
        cv2.destroyAllWindows()

# Call Main function
Main()


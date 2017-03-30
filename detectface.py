import io
import cv2
import numpy
import time
import patternrecognition as pr
import sys

grayscales=1.5
ttl=20
totalcameras=2
#Faces array
rosto=[]
cap=[]
# Find how many cameras are connected
cameras=0
for cameras in range(totalcameras):
    cap.append(cv2.VideoCapture(cameras))
    print(cap[cameras])
if cameras is 0:
    print("No camera found")
    sys.exit()
cameras+=1
print(str(cameras)+" cameras found.\n")
#Opens cam for bright adjustment
tts = 1
timestart = time.time()
timeend = timestart+tts
while timeend>time.time():
    for i in range(cameras):
        ret,frame = cap[i].read()

# Defining program time
timestart = time.time()
timeend = timestart+ttl

#Starting windows Thread
cv2.startWindowThread()
for i in range(cameras):
    cv2.namedWindow("Cam "+str(i))
    ret,frame=cap[i].read()
    cv2.imshow("Cam "+str(i),frame)


face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
counter=0
while timeend>time.time():
    for i in range(cameras):
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
                print("Felipe detectado")
                vetor=pr.calculaLBP(rosto[0],2,2)
                cv2.putText(frame,"Felipe", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            else:
                vetor2=pr.calculaLBP(rosto[counter],2,2)
                distancia=pr.distanciaEuclidiana(vetor,vetor2)
                if distancia<0.1:
                    cv2.putText(frame,"Felipe", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                else:
                    cv2.putText(frame,"Nao sei", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                print("Distancia: "+str(distancia))
            cv2.imshow("Cam "+str(i),frame)
            counter+=1
cv2.imwrite('result.jpg',npaux)
print(len(rosto))

import io
import cv2
import numpy
import time
import patternrecognition as pr
grayscales=1.5
cap = cv2.VideoCapture(1)
rosto=[]
#cap.set(4,800)
#cap.set(5,600)
#cap.set(6,24)
ret,frame = cap.read()

ttl = 10
timestart = time.time()
timeend = timestart+ttl

cv2.startWindowThread()
cv2.namedWindow("LiveDetect")
cv2.imshow("LiveDetect",frame)
rosto1=frame.copy()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
counter=0
while timeend>time.time():
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, grayscales, 5)
    #print("Found "+str(len(faces))+" face(s)")
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        npaux=gray[x:x+w,y:y+h]
        #rostoaux=cv2.cv.fromarray(gray[x:y,x+w:y+h])
        rosto.append(npaux)
        cv2.imshow("LiveDetect",npaux)
        counter+=1
    #rostoaux=cv2.cv.fromarray(gray[x:w,y:h])
    #cv2.imshow("LiveDetect",rostoaux)

print(len(rosto))
cv2.imwrite('result.jpg',rosto.pop())
#cv2.imshow('rosto1',rosto1)
#cvmat=cv2.cv.fromarray(frame)

#vetor=pr.calculaLBP(rosto[0],2,2)
#vetor2=pr.calculaLBP(rosto[1],2,2)
#print(pr.distanciaEuclidiana(vetor,vetor2))
#cv2.waitKey(0)
print(len(rosto))

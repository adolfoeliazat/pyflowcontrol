import io
import cv2
import numpy
import time

grayscales=1.1
cap = cv2.VideoCapture(0)
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

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while timeend>time.time():
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, grayscales, 5)
    #print("Found "+str(len(faces))+" face(s)")
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("LiveDetect",frame)
cv2.imwrite('result.jpg',frame)

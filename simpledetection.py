import io
import cv2
import numpy
import time
import patternrecognition as pr
import sys

grayscales=2
totalcameras=1
#Faces array
rosto=[]
cap=[]
# Find how many cameras are connected
cameras=0
for i in range(totalcameras):
    cap.append(cv2.VideoCapture(i))
    print(cap[i])
    cameras+=1
if cameras is 0:
    print("No camera found")
    sys.exit()

print(str(cameras)+" cameras found.\n")
print("Hit 'q' to quit...\n")

#	Warming cams
#Opens cam for bright adjustment
warm_time = 2 #secs
warm_start = time.time()
warm_end = warm_start+warm_time
while warm_end>time.time():
    for i in range(cameras):
        ret,frame = cap[i].read()

#Starting windows Thread
#cv2.startWindowThread()
for i in range(cameras):
    cv2.namedWindow("Cam "+str(i))
    ret,frame=cap[i].read()
    cv2.imshow("Cam "+str(i),frame)

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
while True:
    for i in range(cameras):
        ret,frame = cap[i].read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, grayscales, 5)
        #print("Found "+str(len(faces))+" face(s)")
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+h,y+w),(255,0,0),2)
        cv2.imshow("Cam "+str(i),frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
for i in range(cameras):
    cap[i].release()
    cv2.destroyAllWindows()

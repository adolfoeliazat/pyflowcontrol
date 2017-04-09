#!/usr/bin/env python3
import cv2
import RPi.GPIO as gpio
from time import sleep

#Program adjustable variables
totalcameras = 1
PIR_sigpin = 11

#Initializing Capture
cap=[]
for i in range(totalcameras) :
    cap.append(cv2.VideoCapture(i))
    cv2.namedWindow("Cam "+str(i))
    ret,frame = cap[i].read()
    cv2.imshow("Cam "+str(i),frame)
    rows,cols,channels = frame.shape

#Setting PIR output
gpio.setmode(gpio.BOARD)
gpio.setup(PIR_sigpin,gpio.IN)

#Capturing frames and showing in different windows
print("Hit 'q' to quit...")
while(True):
    presence = gpio.input(PIR_sigpin)
    if presence == 1:
        for i in range(totalcameras):
            ret,frame=cap[i].read()
            cv2.imshow("Cam "+str(i),frame)
    else:
        cv2.putText(frame,"No presence",(cols/2-200,rows/2-10),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
        cv2.imshow("Cam 0",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing Memory
for i in range(totalcameras):
    cap[i].release()
cv2.destroyAllWindows()

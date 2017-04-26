#!/usr/bin/env python3
import io
import cv2
import numpy
import time
import patternrecognition as pr
import sys
import RPi.GPIO as gpio
from PIL import Image
import zbarlight

#Program adjustable variables
grayscales = 1.7
totalcameras = 2
holdtime = 5
PIR_sigpin = 11
#presence = 0

# Program definitions
holdend = time.time()
hold = False
rosto=[]
cap=[]
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#Setting PIR output
gpio.setmode(gpio.BOARD)
gpio.setup(PIR_sigpin,gpio.IN)

# Defining captures array
for i in range(totalcameras):
    cap.append(cv2.VideoCapture(i))

# Opening windows and warming up cams
for i in range(totalcameras):
    cv2.namedWindow("Cam "+str(i))
warm_time = 5 #secs
warm_end = time.time()+warm_time
while warm_end>time.time():
    for i in range(totalcameras):
        ret,frame = cap[i].read()
        rows,cols,channels = frame.shape
        cv2.putText(frame,"WARMING UP", (cols/2-200,rows/2-10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow("Cam "+str(i),frame)

# Main Loop
print("Hit 'q' to quit...\n")
while True:
    presence = gpio.input(PIR_sigpin)
    if presence == 1 or hold == True:
        for i in range(totalcameras):
            ret,frame = cap[i].read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, grayscales, 5)
            pil = Image.fromarray(gray)
            codes = zbarlight.scan_codes('qrcode',pil)
            if codes is not None:
                print('QR codes: %s'+codes)
            #print("Found "+str(len(faces))+" face(s)")
            if len(faces) > 0:
                hold = True
                holdend = time.time() + holdtime
            else:
                if presence == 0:
                    if time.time() <= holdend:
                        cv2.putText(frame,"HOLDING...", (cols/2-200,rows/2-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255))
                    else:
                        hold = False
                        cv2.putText(frame,"HOLD ENDED.", (cols/2-200,rows/2-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+h,y+w),(255,0,0),2)
            cv2.imshow("Cam "+str(i),frame)
    else:
        cv2.putText(frame,"No presence", (cols/2-200,rows/2-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
        cv2.imshow("Cam "+str(i),frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing Memory
for i in range(totalcameras):
    cap[i].release()
cv2.destroyAllWindows()

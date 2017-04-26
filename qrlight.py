# /usr/bin/python3
import io
import cv2
import numpy
import time
import sys
from PIL import Image
import zbarlight

# Program variables
totalcameras = 2
cap = []
presence = 1

# Defining captures array
for i in range(totalcameras):
    cap.append(cv2.VideoCapture(i))
    cv2.namedWindow("Cam "+str(i))
#Main Loop
print("Hit 'q' to quit...\n")
while True:
    for i in range(totalcameras):
        ret,frame = cap[i].read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        pil = Image.fromarray(gray)
        codes = zbarlight.scan_codes('qrcode',pil)
        if codes is not None:
            print('QR codes: %s' % codes)
        cv2.imshow("Cam "+str(i),frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing Memory
for i in range(totalcameras):
    cap[i].release()
cv2.destroyAllWindows()

#!/usr/bin/env python3
import io
import cv2
import numpy
import time
import patternrecognition as pr
import sys
import threading
#import RPi.GPIO as gpio
from PIL import Image
import zbarlight
from pymongo import MongoClient
from bson.objectid import ObjectId
import ast

# MongoDB definitions
MONGO_HOST = "roscoche.com"
MONGO_PORT = 27017
MONGO_USER = "roscoche"
MONGO_DB = "sentinela"
MONGO_PASS = input("Input MongoDB password for database {}: ".format(MONGO_DB))
connection = MongoClient(MONGO_HOST,MONGO_PORT)
db = connection[MONGO_DB]
db.authenticate(MONGO_USER, MONGO_PASS)
usuarios = db.usuarios.find()
for person in usuarios:
    print(person)

# Program adjustable variables
grayscales = 1.7
totalcameras=2
PIR_sigpin = 11
holdtime = 5

# Program definitions
holdend = time.time()
similarity_threshold = 0.08
rosto=[]
cap=[]
hold = False
counter = 0
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

#Setting PIR output
#gpio.setmode(gpio.BOARD)
#gpio.setup(PIR_sigpin,gpio.IN)

# Defining captures array
for i in range(totalcameras):
    cap.append(cv2.VideoCapture(i))

# Opening windows and warming up cams
for i in range(totalcameras):
    cv2.namedWindow("Cam "+str(i))
warm_time = 2 #secs
warm_end = time.time()+warm_time
while warm_end>time.time():
    for i in range(totalcameras):
        ret,frame = cap[i].read()
        rows,cols,channels = frame.shape
        cv2.putText(frame,"WARMING UP", (int(cols/2-200),int(rows/2-10)), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow("Cam "+str(i),frame)
        

# Main Loop
print("Hit 'q' to quit...\n")
presence=1
PersonName = 'Person 1'
ExpectedPerson = ''
while True:
    #presence = gpio.input(PIR_sigpin)
    if presence == 1 or hold == True:
        for i in range(totalcameras):
            ret,frame = cap[i].read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            pil = Image.fromarray(gray)
            codes = zbarlight.scan_codes('qrcode',pil)
            if codes is not None:
                print('QR code: '+str(codes))
                print(str(codes[0])[2:-1])
                pessoa = db.usuarios.find_one({'_id': ObjectId(str(codes[0])[2:-1])})
                ExpectedCaracteristics = db.usuarios.find_one({'_id': ObjectId(str(codes[0])[2:-1])},{'caracteristicas':1})
                if pessoa is not None:
                    print(pessoa['nome'])
                    ExpectedPerson = pessoa['nome']
                    ExpectedCaracteristics = ast.literal_eval(pessoa['caracteristicas'])
                    carac1 = ExpectedCaracteristics['c1']
                    print('Caracteristicas: ' , carac1)
                    print('LENGTH: ',str(len(ExpectedCaracteristics)))
                    #print('
                    if 0 <= len(ExpectedCaracteristics) :
                        #dbcarac[len(dbcarac)]=str(dbcarac) + ':' + str(vetor2)
                        stringcarac="{{'{}':{}}}".format('c1',str(vetor2))
                        vetor=ExpectedCaracteristics['c1']
                        print("stringcarac " + stringcarac)
                        print("vetor: ",vetor)
                        #print('Nova carac: ' , dbcarac)
                        #db.usuarios.update({'_id': ObjectId(str(codes[0])[2:-1])},{'$set': {'caracteristicas':stringcarac}})
                        print('testando')
                else:
                    print("Person not found on database")
            faces = face_cascade.detectMultiScale(gray, grayscales, 5)
            #print("Found "+str(len(faces))+" face(s)")
            if len(faces) > 0:
                hold = True
                holdend = time.time() + holdtime
            else:
                if presence == 0:
                    if time.time() <= holdend:
                        cv2.putText(frame,"HOLDING...", (int(cols/2-200),int(rows/2-50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255))
                    else:
                        hold = False
                        cv2.putText(frame,"HOLD ENDED.", (int(cols/2-200),int(rows/2-50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0))
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+h,y+w),(255,0,0),2)
                npaux=gray[y:y+h,x:x+w]
                #print("X: {}\tY: {}\tW: {}\tH: {}".format(x,y,w,h))
                #rostoaux=cv2.cv.fromarray(gray[x:y,x+w:y+h])
                rosto.append(npaux)
                if counter == 0:
                    print("Person detected")
                    vetor=pr.calculaLBP(rosto[0],2,2)
                    cv2.putText(frame,PersonName, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                    stringcarac="{{'{}':{}}}".format('c1',str(vetor))
#                    db.usuarios.update({'nome': 'Felipe'},{'$set': {'caracteristicas':stringcarac}})
                else:
                    vetor2=pr.calculaLBP(rosto[counter],2,2)
                    distancia=pr.distanciaEuclidiana(vetor,vetor2)
                    if distancia<similarity_threshold:
                        cv2.putText(frame,ExpectedPerson, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                    else:
                        cv2.putText(frame,"Non authorized person", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                    print("Distancia: "+str(distancia))
                counter += 1
        cv2.imshow("Cam "+str(i),frame)
    else:
        cv2.putText(frame,"No presence", (int(cols/2-200),int(rows/2-10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255))
        cv2.imshow("Cam "+str(i),frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing Memory
for i in range(totalcameras):
    cap[i].release()
cv2.destroyAllWindows()

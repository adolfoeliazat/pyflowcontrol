import io
import cv2
import numpy
import time
import patternrecognition as pr
import numpy as np
import cv2
from matplotlib import pyplot as plt

def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default

def lbp_basic():

    img = cv2.imread('result.jpg', 0)
    transformed_img =  cv2.imread('result.jpg', 0)


    for x in range(0, len(img)):
        for y in range(0, len(img[0])):
            center        = img[x,y]
            top_left      = get_pixel_else_0(img, x-1, y-1)
            top_up        = get_pixel_else_0(img, x, y-1)
            top_right     = get_pixel_else_0(img, x+1, y-1)
            right         = get_pixel_else_0(img, x+1, y )
            left          = get_pixel_else_0(img, x-1, y )
            bottom_left   = get_pixel_else_0(img, x-1, y+1)
            bottom_right  = get_pixel_else_0(img, x+1, y+1)
            bottom_down   = get_pixel_else_0(img, x,   y+1 )

            values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
                                          bottom_down, bottom_left, left])

            weights = [1, 2, 4, 8, 16, 32, 64, 128]
            res = 0
            for a in range(0, len(values)):
                res += weights[a] * values[a]

            transformed_img.itemset((x,y), res)

        print (x)

    #cv2.imshow('image', img)
    cv2.imshow('Imagem LBP', transformed_img)
    cv2.waitKey(0)

    #hist,bins = np.histogram(img.flatten(),256,[0,256])

    #cdf = hist.cumsum()
   # cdf_normalized = cdf * hist.max()/ cdf.max()

    #plt.plot(cdf_normalized, color = 'b')
    #plt.hist(transformed_img.flatten(),256,[0,256], color = 'r')
    #plt.xlim([0,256])
    #plt.legend(('cdf','histogram'), loc = 'upper left')
    #plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 1

grayscales=1.5
cap = cv2.VideoCapture(0)
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

lbp_basic()





import cv2

cap=cv2.VideoCapture(1)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,480)
cv2.startWindowThread()
cv2.namedWindow("BGR")
#cv2.namedWindow("HSV")
#cv2.namedWindow("H")
#cv2.namedWindow("S")
#cv2.namedWindow("V")

while True:
    ret,frame=cap.read()
    #hsv=cv2.cvtColor(frame,cv2.cv.CV_BGR2HSV)
    #h=hsv[:,:,0]
    #s=hsv[:,:,1]
    #v=hsv[:,:,2]
    cv2.imshow("BGR",frame)
    #cv2.imshow("HSV",hsv)
    #cv2.imshow("H",h)
    #cv2.imshow("S",s)
    #cv2.imshow("V",v)

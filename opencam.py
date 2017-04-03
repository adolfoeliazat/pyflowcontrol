import cv2

totalcameras=2

#Initializing Capture
cap=[]
for i in range(totalcameras) :
    cap.append(cv2.VideoCapture(i))
    cv2.namedWindow("Cam "+str(i))
print("Hit 'q' to quit...")

#Capturing frames and showing in different windows
while(True):
    for i in range(totalcameras):
        ret,frame=cap[i].read()
        cv2.imshow("Cam "+str(i),frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release memory
for i in range(totalcameras):
    cap[i].release()
cv2.destroyAllWindows()

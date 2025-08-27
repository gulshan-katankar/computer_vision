import cv2
import mediapipe as mp
import time
import handtrackingmodules as htm

ptime = 0
ctime = 0
cap = cv2.VideoCapture(0) #this opens the default webcam (index 0)
detector = htm.handDetector()

while True : #starts frame by frame video processing
    success, img = cap.read() #reads a frame from the webcam 
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0 :
        print(lmList[4])

    ctime = time.time() #current time
    fps = 1/(ctime-ptime) #fps formula
    ptime = ctime   #update previous time 

    cv2.putText(
        img,
        str(int(fps)),
        (10,70), #position
        cv2.FONT_HERSHEY_PLAIN, #font
        3,
        (255,0,255), #color
        3 #thickness
    )

    cv2.imshow("image", img)
    cv2.waitKey(1)
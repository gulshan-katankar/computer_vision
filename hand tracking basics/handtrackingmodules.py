import cv2
import mediapipe as mp
import time # to check the frame rate




class handDetector() :

    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        

        self.mphands = mp.solutions.hands #give access to hand tracking sol
        self.hands = self.mphands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        ) #creates an instance to detect hands 
        # this sets up a model that detects and tracks up to 2 hands in the image 

        self.mpdraw = mp.solutions.drawing_utils


    def findHands(self, img, draw = True):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks :
            for handLms in self.results.multi_hand_landmarks :
                if draw :
                    self.mpdraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)

        return img
                

    def findPosition(self, img, handNo = 0, draw = True) :
            
            lmList = []

            if self.results.multi_hand_landmarks :
                myHand = self.results.multi_hand_landmarks[handNo]

                for id, lm in enumerate(myHand.landmark):
                    # print(id, lm) #these are the ratios of image we need to multiply width and height for getting the pixel vals
                    h, w, c = img.shape #height width and the channels
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(id, cx, cy) #prints the id along with the landmarks position pixels
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx,cy), 7, (255,0,0), cv2.FILLED)

            return lmList        



def main() :
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0) #this opens the default webcam (index 0)
    detector = handDetector()

    while True : #starts frame by frame video processing
        success, img = cap.read() #reads a frame from the webcam 
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
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




if __name__ == "__main__" :
    main()
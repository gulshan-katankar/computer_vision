import cv2
import mediapipe as mp

cap = cv2.VideoCapture(r'C:\Users\gulsh\OneDrive\Desktop\computer-vision\pose estimation\PoseVideos\4.mp4')

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
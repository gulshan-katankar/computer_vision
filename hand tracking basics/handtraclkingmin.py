# import cv2 # used for capturing video, image processig and displaying frames 
# import mediapipe as mp #a google library for ml based solutions like face detection hand tracking etc
# import time # to check the frame rate

# cap = cv2.VideoCapture(0) #this opens the default webcam (index 0)

# mphands = mp.solutions.hands #give access to hand tracking sol
# hands = mphands.hands() #creates an instance to detect hands 
# # this sets up a model that detects and tracks up to 2 hands in the image 

# while True : #starts frame by frame video processing
#     success, img = cap.read() #reads a frame from the webcam 
#     imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgrgb)
#     # print(results.multi_hand_landmarks)

#     if results.multi_hand_landmarks :
#         for handLms in results.

#     cv2.imshow("image", img)
#     cv2.waitKey(1)

import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
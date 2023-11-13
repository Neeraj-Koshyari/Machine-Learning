import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import cvzone
import random
import time


# Webcam frame
cap = cv2.VideoCapture(0)
cap.set(3, 1280)		#width
cap.set(4, 720)			#height


#Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)


# Find Function
# x is the raw distance y is the value in cm
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x,y,2)		#second order polynomial function to find cofficient a,b,c
# y = ax^2 + bx+c


# Game Variables
cx, cy = 250, 250       #initial button position
color = (0,0,255)       #initial button color
counter = 0             #tell wheter the button is pressed or just the hand is in the frame( < 60 distance)
score = 0               #initial score
timeStart = time.time()
totalTime = 22


# Loop
while True:
 success, img=cap.read()
 img = cv2.flip(img ,1)
 
 if time.time()-timeStart < totalTime:
  hands, img = detector.findHands(img,draw=False)
 
  if hands:
   lmList = hands[0]['lmList']
   x, y, w, h = hands[0]['bbox']			#bounding box x-axes, y-axes, width and height
  
   x1,_,y1 = lmList[5]				#index fingure start point
   x2,_,y2 = lmList[17]				#pinky fingure start point
   
   distance = math.sqrt((y2-y1)**2 + (x2-x1)**2)		#to resolve hand rotation problem
  
   A, B, C =coff
   distanceCM = A * distance**2 + B * distance + C			# equation, y = ax^2 + bx+c		where x is distance in centimeter (cm)
   #print(distanceCM, distance)
  
  
   # Hand is in the range or not
   if distanceCM < 60 and x < cx < x + w and y < cy < y + h:
    counter = 1
  
  
   #display cordinate in Hand
   cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,255), 3)			#to show the rectangle in image of hand
   cvzone.putTextRect(img, f'{int(distanceCM)} cm', (x+8, y-10))		    #show distance in image of hand
   
  
  # if hand in range change color and reset after 4 sec
  if counter:
      counter+=1
      color = (0,255,0)
      if counter == 4:
       cx = random.randint(100, 1100)    #random positon for button
       cy = random.randint(100, 600)
       color = (0,0,255)
       score+=1                          #increase score after one button press
       counter = 0                       #reset counter
   
   
  # Draw Button
  cv2.circle(img, (cx, cy), 32, (0,0,0), cv2.FILLED)
  cv2.circle(img, (cx, cy), 28, (255,255,255), cv2.FILLED)
  cv2.circle(img, (cx, cy), 20, color, cv2.FILLED)
  
  
  # Game Head Up Display
  cvzone.putTextRect(img, f'Time: {int(totalTime-(time.time()-timeStart))}', (1000, 75), scale=3, offset=20)        #Time bar
  cvzone.putTextRect(img, f'Score: {str(score).zfill(2)}', (60, 75), scale=3, offset=20)         #Points bar
  
 else:
  cvzone.putTextRect(img, 'Game Over', (400, 400), scale=5, offset=30, thickness=7)
  cvzone.putTextRect(img, f'Your Score: {str(score).zfill(2)}', (440, 500), scale=3, offset=20)
  cvzone.putTextRect(img, 'Press \'r\' to restart', (460, 575), scale=2, offset=10)
  
  
 cv2.imshow("Image",img)
 key = cv2.waitKey(1)
 
 if key == ord('r'):
  timeStart = time.time()
  score = 0

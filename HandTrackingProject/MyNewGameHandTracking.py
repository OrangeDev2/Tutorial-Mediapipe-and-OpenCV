import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm


# Access and assign web camera to 'capture'
capture = cv2.VideoCapture(0)                                          # Option 1: Web camera

# capture = cv2.VideoCapture('test.mp4')                               # Option 2: Play video from a file

# img = cv2.imread(cv2.samples.findFile("test.png"))                   # Option 3: Display image

# Previous and current time
pTime = 0
cTime = 0


detector = htm.handDetector()                           # handDetector class as detector within HandTrackingModule.py

# Create a true loop to show live picture in new window screen
# Image is shown with a call to cv2::imshow function
while True:
    success, img = capture.read()
    img = detector.findHands(img)                         # pass img to findHands() method within detector = handDetectors() class.
    lmList = detector.findPosition(img, draw=False)                   # return value from method call assigned to lmList
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (10, 120), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Display window", img)
    cv2.waitKey(1)
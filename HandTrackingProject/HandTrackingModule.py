import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands = 2, complexity = 1, detectionCon =0.5, trackCon=0.5):                  # Initiative default parameters and variables for the rest of program
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity

        # Mediapipe solutions for drawing and hands.
        self.mpDraw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findHands(self, img, draw = True):                                                                       # A method to draw hand landmarks
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)


        if results.multi_hand_landmarks:                                                            # Draw landmarks and connections by extracting information from results.
            for handLms in results.multi_hand_landmarks:
                # print(results.multi_hand_landmarks)

                if draw:
                    self.mpDraw.draw_landmarks(
                            img,                                                                 # draw on original image
                            handLms,                                                             # looping landmarks (21 points) #0, 1, 2... for hands
                            self.mpHands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        return img

                #for id, lm in enumerate(handLms.landmark):                                          # access landmark from results.multi_hand_landmarks List
                #    #print("id: " + str(id) + "\n" + "landmark position (x, y, z): " + "\n" + str(lm))   # id -> 0-20, lm -> x: 1 y: 2 z: 3
                #    #print(handLms.landmark)
                #    h, w, c = img.shape
                #    cx, cy = int(lm.x*w), int(lm.y*h)
                #    #print(id, cx, cy)
                #    if id == 4:                                                                     # Example, id == 4 then (cx, cy) extracted at 4 only
                #    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)                          # Draw a filled circle at (cx, cy) on the image

def main():
    # Access and assign web camera to 'capture'
    capture = cv2.VideoCapture(0)                                          # Option 1: Web camera

    # capture = cv2.VideoCapture('test.mp4')                               # Option 2: Play video from a file

    # img = cv2.imread(cv2.samples.findFile("test.png"))                   # Option 3: Display image

    # Previous and current time
    pTime = 0
    cTime = 0


    detector = handDetector()

    # Create a true loop to show live picture in new window screen
    # Image is shown with a call to cv2::imshow function
    while True:
        success, img = capture.read()
        img = detector.findHands(img)                         # pass img to findHands() method within detector = handDetectors() class.

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "FPS: " + str(int(fps)), (10, 120), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)

        cv2.imshow("Display window", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()
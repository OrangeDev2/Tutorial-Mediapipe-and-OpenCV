import cv2
import mediapipe as mp
import time

# Access and assign web camera to 'capture'
capture = cv2.VideoCapture(0)                                         # Option 1: Web camera

# Same as above but playing video from a file
#capture = cv2.VideoCapture('test.mp4')                               # Option 2: Play video from a file

# Same as above but single image with a frame rate
#img = cv2.imread(cv2.samples.findFile("test.png"))                   # Option 3: Display image

# adding Mediapipe solutions for drawing and hands.
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()

mp_drawing_styles = mp.solutions.drawing_styles

# Previous and current time
pTime = 0
cTime = 0

# Create a true loop to show live picture in new window screen
# Image is shown with a call to cv2::imshow function
while True:
    success, img = capture.read()                                      # Comment this out if you want to process an image from file instead of a video.

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)                                # outputs whenever mediapipe detect hands

    if results.multi_hand_landmarks:                                    # Draw landmarks and connections by extracting information from results.
        for handLms in results.multi_hand_landmarks:

            for id, lm in enumerate(handLms.landmark):
                #print("id: " + str(id) + "\n" + "landmark position (x, y, z): " + "\n" + str(lm))   # id -> 0-20, lm -> x: 1 y: 2 z: 3
                #print(handLms.landmark)

                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                if id == 4:                                                     # Example, id == 4 then (cx, cy) extracted at 4 only
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)       # Draw a filled circle at (cx, cy) on the image

            mpDraw.draw_landmarks(
                img,                                                    # draw on original image
                handLms,                                                # looping landmarks (21 points) #0, 1, 2... for hands
                mpHands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (10, 120), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,0), 3)

    cv2.imshow("Display window", img)
    cv2.waitKey(1)                                                      # IMPORTANT: Set 0 for image process.  Set 1 for video process


import cv2
import mediapipe as mp
import time

# Access and assign web camera to 'capture'
capture = cv2.VideoCapture(0)

# adding Mediapipe solutions for drawing and hands.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Create a true loop to show live picture in new window screen
# Image is shown with a call to cv2::imshow function

while True:
    success, img = capture.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # outputs whenever mediapipe detect hands
    #print(results.multi_hand_landmarks)

    print(hands.process(imgRGB))

    # runs if results.multi_hand_landmarks or hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    cv2.imshow("Display window", img)
    # Set 0 for image process.  Set 1 for video process
    cv2.waitKey(1)


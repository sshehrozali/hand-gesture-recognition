import cv2      # Import OpenCV Library
import mediapipe as mp

video = cv2.VideoCapture(0)     # Open default Webcam
mp_draw = mp.solutions.drawing_utils    # Create drawing lines
mp_hands = mp.solutions.hands           # Create Hand drawing solution

with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:

    while True:
        ret, img = video.read()     # Read video

        # Create Handpose (Mediapipe)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # Converts img to RGB
        img.flags.writeable = False
        output = hands.process(img) # Returns mapped img
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)      # Converts RGB to img -> (human readable vision)

        # Creates Hand posture
        if output.multi_hand_landmarks:

            # Iterate over each landmark in hand landmarks i.e. small dots mapped on hand img
            for hand_landmark in output.multi_hand_landmarks:

                # Draw line and connect each dots to create hand posture 
                mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)


        cv2.imshow("Frame", img)    # Show video frame
        
        key = cv2.waitKey(1)        # Breaks the loop if 'q' is pressed
        if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
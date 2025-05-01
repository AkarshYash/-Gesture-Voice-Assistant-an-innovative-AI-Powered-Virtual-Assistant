import cv2
import mediapipe as mp
import pyttsx3
import webbrowser
import os
import time
import subprocess

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.say("Hello Sir, How can I help you in today's tasks?")
engine.runAndWait()   

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Load Haar cascade for eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

last_eye_open_time = time.time()
eye_closed_alert_given = False

current_process = None  # Track the current opened process

# Function to check if a hand is forming a closed fist
def is_closed_fist(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_bases = [6, 10, 14, 18]
    closed_fingers = 0
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[base].y:
            closed_fingers += 1
    return closed_fingers == 4

# Function to handle window closing/opening logic
def open_new_app(command, message):
    global current_process
    if current_process:
        current_process.terminate()
        time.sleep(1)

    engine.say(message)
    engine.runAndWait()

    if command.startswith("http"):
        webbrowser.open(command)
    else:
        current_process = subprocess.Popen(command, shell=True)

# Function to detect hand gestures and open corresponding apps
def open_application(fingers, hand_label):
    if hand_label == "Right":
        if fingers == [0, 1, 0, 0, 0]:
            open_new_app("https://www.youtube.com/", "Opening YouTube")
        elif fingers == [0, 1, 1, 0, 0]:
            open_new_app("https://www.hotstar.com/", "Opening Hotstar")
        elif fingers == [0, 1, 1, 1, 0]:
            open_new_app("start microsoft.windows.camera:", "Opening Camera")
        elif fingers == [0, 1, 1, 1, 1]:
            open_new_app("start msedge", "Opening Microsoft Edge")
        elif fingers == [1, 1, 1, 1, 1]:
            open_new_app("notepad", "Opening Notepad")

# Main loop
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    eyes_detected = False
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) > 0:
            eyes_detected = True

    if eyes_detected:
        last_eye_open_time = time.time()
        eye_closed_alert_given = False
    else:
        if time.time() - last_eye_open_time > 5:
            if not eye_closed_alert_given:
                engine.say("Alert! Please open your eyes.")
                engine.runAndWait()
                eye_closed_alert_given = True

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                lmList.append((int(lm.x * w), int(lm.y * h)))

            fingers = []

            if handLms.landmark[4].x < handLms.landmark[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            for tip in [8, 12, 16, 20]:
                if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            hand_label = "Right"  # Assuming one hand for simplicity

            open_application(fingers, hand_label)

    cv2.imshow("Virtual Assistant", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
















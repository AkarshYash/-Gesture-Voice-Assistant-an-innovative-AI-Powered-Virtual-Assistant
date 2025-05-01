
import cv2
import mediapipe as mp
import pyttsx3
import tkinter as tk
from tkinter import messagebox
import webbrowser
import os
import time

# Text-to-Speech Engine
engine = pyttsx3.init()

# MediaPipe Initialization
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Tkinter GUI Window
def start_assistant():
    engine.say("Welcome back! How can I help you today?")
    engine.runAndWait()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture & Voice Assistant", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def exit_program():
    confirm = messagebox.askyesno("Exit", "Do you really want to exit?")
    if confirm:
        root.destroy()


# GUI Window Layout
root = tk.Tk()
root.title("Gesture & Voice Assistant")
root.geometry("400x700")
root.configure(bg="#222222")

label = tk.Label(root, text="Gesture & Voice Assistant", font=("Arial", 16), fg="white", bg="#222222")
label.pack(pady=20)

start_button = tk.Button(root, text="Start Assistant", font=("Arial", 14), bg="#4CAF50", fg="white", command=start_assistant)
start_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", font=("Arial", 14), bg="#f44336", fg="white", command=exit_program)
exit_button.pack(pady=10)

root.mainloop()

 

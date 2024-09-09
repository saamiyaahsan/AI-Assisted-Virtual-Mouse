from turtle import ht
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import matplotlib.pyplot as plt
from IPython.display import Image

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Smoothing factor for cursor movement
smoothening = 9
plocx, plocy = 0, 0
clocx, clocy = 0, 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            index_x, index_y = None, None
            thumb_x, thumb_y = None, None

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip point number is 8
                    cv2.circle(img=frame, center=(x, y),
                               radius=15, color=(0, 255, 255))
                    index_x = (screen_width / frame_width) * x
                    index_y = (screen_height / frame_height) * y

                if id == 4:  # Thumb tip point number is 4
                    cv2.circle(img=frame, center=(x, y),
                               radius=15, color=(255, 0, 255))
                    thumb_x = (screen_width / frame_width) * x
                    thumb_y = (screen_height / frame_height) * y

            if index_x is not None and index_y is not None and thumb_x is not None and thumb_y is not None:
                # Calculate distance between index and thumb fingers
                distance = np.sqrt((index_x - thumb_x) **
                                   2 + (index_y - thumb_y)**2)

                if distance < 50:  # Zoom in if fingers are close
                    pyautogui.hotkey('command', '+')
                elif distance > 100:  # Zoom out if fingers are far apart
                    pyautogui.hotkey('command', '-')
    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)

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

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip point number is 8
                    cv2.circle(img=frame, center=(x, y),
                               radius=15, color=(0, 255, 255))
                    index_x = (screen_width / frame_width) * x
                    index_y = (screen_height / frame_height) * y
                    clocx = plocx + (index_x - plocx) / smoothening
                    clocy = plocy + (index_y - plocy) / smoothening
                    pyautogui.moveTo(clocx, clocy)
                    plocx, plocy = clocx, clocy

                if id == 12:  # Middle finger tip point number is 12
                    cv2.circle(img=frame, center=(x, y),
                               radius=15, color=(255, 0, 255))
                    middle_x = (screen_width / frame_width) * x
                    middle_y = (screen_height / frame_height) * y
                    if middle_y < index_y:  # Scroll up
                        pyautogui.scroll(10)
                    elif middle_y > index_y:  # Scroll down
                        pyautogui.scroll(-10)

    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)

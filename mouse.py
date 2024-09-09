
import cv2
import numpy as np
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
screen_width, screen_height = pyautogui.size()

# Smoothness factor for cursor movement
SMOOTHNESS_FACTOR = 0.5

# Initialize previous finger positions
previous_x = None

# Function to move the cursor smoothly


def move_cursor_smoothly(x, y, smoothness):
    current_x, current_y = pyautogui.position()
    target_x = int(x * screen_width)
    target_y = int(y * screen_height)
    new_x = current_x + smoothness * (target_x - current_x)
    new_y = current_y + smoothness * (target_y - current_y)
    pyautogui.moveTo(new_x, new_y)

# Function to detect two fingers


def detect_two_fingers(landmarks):
    return len(landmarks) == 2

# Function to detect three fingers


def detect_three_fingers(landmarks):
    return len(landmarks) == 3


# Main loop
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            landmarks = [(lm.x, lm.y) for lm in hand.landmark]
            thumb_x, thumb_y = landmarks[4]  # Thumb tip
            index_x, index_y = landmarks[8]  # Index finger tip
            middle_x, middle_y = landmarks[12]  # Middle finger tip

            # Perform actions based on finger positions
            if detect_two_fingers(landmarks) and abs(thumb_x - index_x) < 0.05 and abs(thumb_y - index_y) < 0.05:
                # Single click when thumb and index finger are joined
                pyautogui.click()
            elif detect_two_fingers(landmarks) and abs(thumb_x - middle_x) < 0.05 and abs(thumb_y - middle_y) < 0.05:
                # Double click when thumb and middle finger are joined
                pyautogui.doubleClick()

            # Move the cursor smoothly using the index finger position
            move_cursor_smoothly(index_x, index_y, SMOOTHNESS_FACTOR)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

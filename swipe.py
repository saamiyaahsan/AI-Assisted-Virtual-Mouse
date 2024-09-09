import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
screen_width, screen_height = pyautogui.size()

# Smoothness factor for cursor movement
SMOOTHNESS_FACTOR = 0.5
# Number of frames to use for the moving average filter
MOVING_AVERAGE_WINDOW = 5
# Delay in seconds for the swipe action
SWIPE_DELAY = 1.0  # Adjust as needed

# Initialize previous finger position
previous_x = None

# Function to move the cursor smoothly


def move_cursor_smoothly(x, y, smoothness):
    current_x, current_y = pyautogui.position()
    target_x = int(x * screen_width)
    target_y = int(y * screen_height)
    new_x = current_x + smoothness * (target_x - current_x)
    new_y = current_y + smoothness * (target_y - current_y)
    pyautogui.moveTo(new_x, new_y)

# Function to check if fingers moved to the right


def fingers_moved_right(previous_x, current_x):
    if previous_x is not None:
        return current_x > previous_x + 0.05  # Adjust sensitivity by changing the value
    return False

# Function to check if fingers moved to the left


def fingers_moved_left(previous_x, current_x):
    if previous_x is not None:
        return current_x < previous_x - 0.05  # Adjust sensitivity by changing the value
    return False


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
            index_x, index_y = landmarks[8]  # Index finger tip

            # Perform actions based on finger positions
            if fingers_moved_right(previous_x, index_x):
                # Perform right screen shift action
                # Example: Ctrl + Right arrow
                pyautogui.hotkey('ctrl', 'right')
                time.sleep(SWIPE_DELAY)  # Introduce delay
            elif fingers_moved_left(previous_x, index_x):
                # Perform left screen shift action
                pyautogui.hotkey('ctrl', 'left')  # Example: Ctrl + Left arrow
                time.sleep(SWIPE_DELAY)  # Introduce delay

            # Store previous finger positions for the next iteration
            previous_x = index_x

            # Move the cursor smoothly using the index finger position
            move_cursor_smoothly(index_x, index_y, SMOOTHNESS_FACTOR)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

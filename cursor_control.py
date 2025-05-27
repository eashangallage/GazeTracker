import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import time
import os

# --- Configuration ---
CALIBRATION_DATA_FILE = 'calibration_data.pkl'
BLINK_THRESHOLD = 0.22  # Adjust this value based on your eye aspect ratio for a blink
BLINK_CONSEC_FRAMES = 3 # Number of consecutive frames below threshold to detect a blink
MOUSE_SMOOTHING_FACTOR = 0.7 # 0.0 (no smoothing) to 1.0 (max smoothing) - Higher value means more smoothing
CLICK_COOLDOWN_SECONDS = 0.5 # Time in seconds to prevent rapid multiple clicks

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Global variables for calibration and state ---
calibration_data = None
blink_frame_counter = 0
last_click_time = 0

# --- Eye Aspect Ratio (EAR) calculation ---
# MediaPipe landmarks for EAR (approximate, based on common interpretations)
# These indices are for the refined_landmarks=True output
# Left eye (user's left eye, image right):
# p1=362, p2=382, p3=381, p4=263, p5=375, p6=374
LEFT_EYE_LANDMARKS = [362, 382, 381, 263, 375, 374]
# Right eye (user's right eye, image left):
# p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two 2D points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR) for a given eye.
    landmarks: MediaPipe normalized landmarks list.
    eye_indices: List of 6 landmark indices for the eye (p1, p2, p3, p4, p5, p6).
    """
    if not landmarks or len(landmarks) < max(eye_indices) + 1:
        return 0.0

    # Get the coordinates of the 6 eye landmarks
    p = []
    for idx in eye_indices:
        p.append(np.array([landmarks[idx].x, landmarks[idx].y]))

    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = euclidean_distance(p[1], p[5]) # p2-p6
    B = euclidean_distance(p[2], p[4]) # p3-p5

    # Compute the euclidean distance between the horizontal eye landmark
    C = euclidean_distance(p[0], p[3]) # p1-p4

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def load_calibration_data():
    """Loads calibration data from the specified file."""
    global calibration_data
    if os.path.exists(CALIBRATION_DATA_FILE):
        try:
            with open(CALIBRATION_DATA_FILE, 'rb') as f:
                calibration_data = pickle.load(f)
            print(f"Calibration data loaded from {CALIBRATION_DATA_FILE}")
            print(f"Calibration Ranges: {calibration_data}")
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            calibration_data = None
    else:
        print(f"Calibration data file '{CALIBRATION_DATA_FILE}' not found.")
        print("Please run 'calibrate.py' first to calibrate your eye movements.")

def get_eye_center_for_mapping(landmarks):
    """
    Calculates the average center of both eyes based on iris landmarks for mapping.
    This function is identical to get_eye_center in calibrate.py to ensure consistency.
    """
    if not landmarks:
        return None

    left_iris_indices = [474, 475, 476, 477]
    right_iris_indices = [469, 470, 471, 472]

    all_iris_landmarks = []
    for idx in left_iris_indices + right_iris_indices:
        if idx < len(landmarks):
            all_iris_landmarks.append(landmarks[idx])

    if not all_iris_landmarks:
        return None

    avg_x = sum([lm.x for lm in all_iris_landmarks]) / len(all_iris_landmarks)
    avg_y = sum([lm.y for lm in all_iris_landmarks]) / len(all_iris_landmarks)

    return avg_x, avg_y

def run_cursor_control():
    """
    Runs the eye-tracking cursor control.
    Loads calibration data, tracks eye movements, moves cursor, and clicks on blink.
    """
    global blink_frame_counter, last_click_time

    load_calibration_data()
    if calibration_data is None:
        print("Exiting: Calibration data is required to run cursor control.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    screen_width = calibration_data['screen_width']
    screen_height = calibration_data['screen_height']

    # Initialize previous mouse position for smoothing
    prev_mouse_x, prev_mouse_y = pyautogui.position()

    print("Starting cursor control. Press 'q' to quit.")
    print("Blink to click.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1) # Mirror the frame

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # --- Gaze Tracking for Cursor Movement ---
                # Get the average eye center using both eyes for mapping
                eye_center = get_eye_center_for_mapping(face_landmarks.landmark)

                if eye_center:
                    eye_x_norm, eye_y_norm = eye_center[0], eye_center[1]

                    # Normalize eye coordinates to a 0-1 range based on calibration
                    # Clamp values to prevent out-of-range issues and ensure mapping within bounds
                    clamped_eye_x = np.clip(eye_x_norm, calibration_data['eye_x_min'], calibration_data['eye_x_max'])
                    clamped_eye_y = np.clip(eye_y_norm, calibration_data['eye_y_min'], calibration_data['eye_y_max'])

                    # Map normalized eye coordinates to screen coordinates
                    mapped_x = np.interp(clamped_eye_x,
                                         [calibration_data['eye_x_min'], calibration_data['eye_x_max']],
                                         [0, screen_width])
                    mapped_y = np.interp(clamped_eye_y,
                                         [calibration_data['eye_y_min'], calibration_data['eye_y_max']],
                                         [0, screen_height])

                    # Apply smoothing to mouse movement
                    current_mouse_x = int(prev_mouse_x * MOUSE_SMOOTHING_FACTOR + mapped_x * (1 - MOUSE_SMOOTHING_FACTOR))
                    current_mouse_y = int(prev_mouse_y * MOUSE_SMOOTHING_FACTOR + mapped_y * (1 - MOUSE_SMOOTHING_FACTOR))

                    pyautogui.moveTo(current_mouse_x, current_mouse_y)
                    prev_mouse_x, prev_mouse_y = current_mouse_x, current_mouse_y

                    # Draw a small circle at the current mapped gaze position on the webcam feed
                    # This helps visualize where the system thinks you're looking
                    frame_h, frame_w, _ = frame.shape
                    gaze_x_on_frame = int(eye_x_norm * frame_w)
                    gaze_y_on_frame = int(eye_y_norm * frame_h)
                    cv2.circle(frame, (gaze_x_on_frame, gaze_y_on_frame), 5, (0, 255, 255), -1) # Yellow dot

                # --- Blink Detection for Click ---
                left_eye_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_LANDMARKS)
                right_eye_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_LANDMARKS)

                # Use the average EAR of both eyes for more robust blink detection
                avg_ear = (left_eye_ear + right_eye_ear) / 2.0

                if avg_ear < BLINK_THRESHOLD:
                    blink_frame_counter += 1
                else:
                    if blink_frame_counter >= BLINK_CONSEC_FRAMES:
                        # Blink detected, perform click if cooldown allows
                        current_time = time.time()
                        if (current_time - last_click_time) > CLICK_COOLDOWN_SECONDS:
                            pyautogui.click()
                            last_click_time = current_time
                            print("Click!")
                    blink_frame_counter = 0 # Reset counter

                # Display EAR on frame for debugging
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Blink Counter: {blink_frame_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Eye Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Cursor control stopped.")

if __name__ == "__main__":
    run_cursor_control()
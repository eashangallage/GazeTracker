import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import time
import os
import random

# --- Configuration ---
# Increased and randomized calibration points for better coverage
CALIBRATION_POINTS = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # Top row
    (0.1, 0.3), (0.5, 0.3), (0.9, 0.3),  # Upper-mid row
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # Middle row
    (0.1, 0.7), (0.5, 0.7), (0.9, 0.7),  # Lower-mid row
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)   # Bottom row
]
CALIBRATION_SAMPLES_PER_POINT = 40  # Number of frames to collect data for each point
CALIBRATION_DELAY_SECONDS = 1.5      # Delay before collecting samples for each point
CALIBRATION_CIRCLE_RADIUS = 20     # Radius of the calibration target circle
CALIBRATION_CIRCLE_COLOR = (0, 255, 0) # Green
CALIBRATION_TEXT_COLOR = (255, 255, 255) # White
CALIBRATION_BG_COLOR = (0, 0, 0) # Black
CALIBRATION_FONT = cv2.FONT_HERSHEY_SIMPLEX
CALIBRATION_DATA_FILE = 'calibration_data.pkl'

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Enables iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Global variables for calibration data ---
# These will store the min/max eye coordinates observed during calibration
# These ranges will be used to map eye movements to screen coordinates
calibration_ranges = {
    'eye_x_min': float('inf'),
    'eye_x_max': float('-inf'),
    'eye_y_min': float('inf'),
    'eye_y_max': float('-inf'),
    'screen_width': pyautogui.size().width,
    'screen_height': pyautogui.size().height
}

def get_eye_center(landmarks):
    """
    Calculates the average center of both eyes based on iris landmarks.
    Using both eyes should provide a more stable and accurate gaze estimate.
    """
    if not landmarks:
        return None

    # MediaPipe iris landmarks for both eyes (from refined_landmarks=True)
    # Left Iris (user's left eye, image right): 474, 475, 476, 477
    # Right Iris (user's right eye, image left): 469, 470, 471, 472
    left_iris_indices = [474, 475, 476, 477]
    right_iris_indices = [469, 470, 471, 472]

    all_iris_landmarks = []
    # Collect landmarks for both irises
    for idx in left_iris_indices + right_iris_indices:
        if idx < len(landmarks):
            all_iris_landmarks.append(landmarks[idx])

    if not all_iris_landmarks:
        return None

    # Calculate the average x and y coordinates of all collected iris landmarks
    avg_x = sum([lm.x for lm in all_iris_landmarks]) / len(all_iris_landmarks)
    avg_y = sum([lm.y for lm in all_iris_landmarks]) / len(all_iris_landmarks)

    return avg_x, avg_y

def run_calibration():
    """
    Runs the eye-tracking calibration process.
    Displays randomized calibration points and collects averaged eye data for mapping.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a full-screen window for calibration
    cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Starting calibration...")
    print("Please keep your head relatively still during calibration.")
    print("Look at the green circle when it appears.")

    # Shuffle calibration points for randomization
    shuffled_calibration_points = list(CALIBRATION_POINTS)
    random.shuffle(shuffled_calibration_points)

    # List to store average eye positions for each calibration point
    all_avg_eye_points_collected = []

    for i, (norm_x, norm_y) in enumerate(shuffled_calibration_points):
        screen_x = int(norm_x * calibration_ranges['screen_width'])
        screen_y = int(norm_y * calibration_ranges['screen_height'])

        collected_eye_x_samples = []
        collected_eye_y_samples = []

        # Display calibration point and instructions
        start_time = time.time()
        while time.time() - start_time < CALIBRATION_DELAY_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1) # Mirror the frame

            # Create a black screen for calibration display
            calibration_display = np.zeros((calibration_ranges['screen_height'], calibration_ranges['screen_width'], 3), dtype=np.uint8)

            # Draw instructions
            text = f"Look at the green circle. Point {i+1}/{len(shuffled_calibration_points)}"
            text_size = cv2.getTextSize(text, CALIBRATION_FONT, 1, 2)[0]
            text_x = (calibration_display.shape[1] - text_size[0]) // 2
            text_y = calibration_display.shape[0] // 2 - 100
            cv2.putText(calibration_display, text, (text_x, text_y), CALIBRATION_FONT, 1, CALIBRATION_TEXT_COLOR, 2, cv2.LINE_AA)

            # Draw calibration circle
            cv2.circle(calibration_display, (screen_x, screen_y), CALIBRATION_CIRCLE_RADIUS, CALIBRATION_CIRCLE_COLOR, -1)

            cv2.imshow('Calibration', calibration_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f"Collecting data for point {i+1}/{len(shuffled_calibration_points)} at ({screen_x}, {screen_y})...")

        for _ in range(CALIBRATION_SAMPLES_PER_POINT):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Process the frame for face mesh detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get the average eye center using both eyes
                    eye_center = get_eye_center(face_landmarks.landmark)

                    if eye_center:
                        collected_eye_x_samples.append(eye_center[0])
                        collected_eye_y_samples.append(eye_center[1])

            # Display the calibration target on a black screen
            calibration_display = np.zeros((calibration_ranges['screen_height'], calibration_ranges['screen_width'], 3), dtype=np.uint8)
            cv2.circle(calibration_display, (screen_x, screen_y), CALIBRATION_CIRCLE_RADIUS, CALIBRATION_CIRCLE_COLOR, -1)
            cv2.imshow('Calibration', calibration_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        if collected_eye_x_samples and collected_eye_y_samples:
            # Take the average of collected samples for this point
            avg_eye_x_for_point = np.mean(collected_eye_x_samples)
            avg_eye_y_for_point = np.mean(collected_eye_y_samples)
            all_avg_eye_points_collected.append((avg_eye_x_for_point, avg_eye_y_for_point))
        else:
            print(f"Warning: No valid eye data collected for point {i+1}. Ensure your face is visible.")

    cap.release()
    cv2.destroyAllWindows()
    print("Calibration complete!")

    if all_avg_eye_points_collected:
        # Calculate the overall min/max from all collected average eye points
        calibration_ranges['eye_x_min'] = min(p[0] for p in all_avg_eye_points_collected)
        calibration_ranges['eye_x_max'] = max(p[0] for p in all_avg_eye_points_collected)
        calibration_ranges['eye_y_min'] = min(p[1] for p in all_avg_eye_points_collected)
        calibration_ranges['eye_y_max'] = max(p[1] for p in all_avg_eye_points_collected)
        print(f"Final Calibration Ranges: {calibration_ranges}")
    else:
        print("Error: No valid eye data collected during calibration. Calibration failed.")
        return

    # Save calibration data
    try:
        with open(CALIBRATION_DATA_FILE, 'wb') as f:
            pickle.dump(calibration_ranges, f)
        print(f"Calibration data saved to {CALIBRATION_DATA_FILE}")
    except Exception as e:
        print(f"Error saving calibration data: {e}")

if __name__ == "__main__":
    run_calibration()

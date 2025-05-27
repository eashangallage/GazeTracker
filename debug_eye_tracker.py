import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import time
import os
import random

# --- Configuration ---
CALIBRATION_DATA_FILE = 'calibration_data.pkl'
DEBUG_POINTS = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
]
DEBUG_SAMPLES_PER_POINT = 60 # Number of frames to collect data and display debug info
DEBUG_DELAY_SECONDS = 2      # Delay before collecting samples for each point
DEBUG_CIRCLE_RADIUS = 20     # Radius of the debug target circle
DEBUG_CIRCLE_COLOR = (0, 255, 255) # Yellow
DEBUG_TEXT_COLOR = (255, 255, 255) # White
DEBUG_BG_COLOR = (0, 0, 0) # Black
DEBUG_FONT = cv2.FONT_HERSHEY_SIMPLEX

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

# --- Global variables for calibration data ---
calibration_data = None

# --- 3D Model Points for Head Pose Estimation ---
# These are approximate 3D model points of a human face, corresponding to MediaPipe landmarks
# The unit doesn't matter as long as it's consistent (e.g., millimeters)
# Source: Based on common facial landmark models for solvePnP
model_points = np.array([
    (0.0, 0.0, 0.0),         # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),    # Chin (landmark 152)
    (-225.0, 170.0, -135.0), # Left eye corner (outer, landmark 33)
    (225.0, 170.0, -135.0),  # Right eye corner (outer, landmark 263)
    (-150.0, -150.0, -125.0),# Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)  # Right mouth corner (landmark 291)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices for the model_points
FACE_MESH_3D_POINTS_INDICES = [1, 152, 33, 263, 61, 291]

def get_eye_center(landmarks):
    """
    Calculates the average center of both eyes based on iris landmarks.
    This function is consistent with calibrate.py and cursor_control.py.
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

def get_head_pose(landmarks, image_shape):
    """
    Estimates head pose (pitch, yaw, roll) using solvePnP.
    """
    if not landmarks:
        return None, None, None

    # 2D image points from MediaPipe landmarks
    image_points = np.array([
        (landmarks[idx].x * image_shape[1], landmarks[idx].y * image_shape[0])
        for idx in FACE_MESH_3D_POINTS_INDICES
    ], dtype=np.float64)

    # Camera internals (approximate, usually estimated from calibration)
    # For a general webcam, these are often approximated.
    # fx, fy: focal lengths
    # cx, cy: principal point (center of image)
    focal_length = image_shape[1]
    center = (image_shape[1]/2, image_shape[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion for simplicity

    # SolvePnP to get rotation and translation vectors
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Decompose projection matrix to get Euler angles (pitch, yaw, roll)
        # pitch: rotation around X-axis (up/down)
        # yaw: rotation around Y-axis (left/right)
        # roll: rotation around Z-axis (tilt)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch = eulerAngles[0, 0]
        yaw = eulerAngles[1, 0]
        roll = eulerAngles[2, 0]

        return pitch, yaw, roll
    return None, None, None

def run_debug_mode():
    """
    Runs the debugging mode for eye-tracking.
    Loads calibration data, displays target points, and prints debug information.
    """
    load_calibration_data()
    if calibration_data is None:
        print("Exiting: Calibration data is required for debugging.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    screen_width = calibration_data['screen_width']
    screen_height = calibration_data['screen_height']

    # Create a full-screen window for debugging display
    cv2.namedWindow('Debug Display', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Debug Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n--- Starting Debug Mode ---")
    print("Look at the yellow circle when it appears.")
    print("Compare 'Mapped Mouse:' with 'Actual Mouse:' and 'Head Pose' for insights.")
    print("Press 'q' to quit at any time.")

    # Shuffle debug points for randomization
    shuffled_debug_points = list(DEBUG_POINTS)
    random.shuffle(shuffled_debug_points)

    for i, (norm_x, norm_y) in enumerate(shuffled_debug_points):
        target_screen_x = int(norm_x * screen_width)
        target_screen_y = int(norm_y * screen_height)

        print(f"\n--- Debug Point {i+1}/{len(shuffled_debug_points)}: Look at ({target_screen_x}, {target_screen_y}) ---")

        start_time = time.time()
        while time.time() - start_time < DEBUG_DELAY_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            debug_display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            text = f"Look at the yellow circle. Point {i+1}/{len(shuffled_debug_points)}"
            text_size = cv2.getTextSize(text, DEBUG_FONT, 1, 2)[0]
            text_x = (debug_display.shape[1] - text_size[0]) // 2
            text_y = debug_display.shape[0] // 2 - 100
            cv2.putText(debug_display, text, (text_x, text_y), DEBUG_FONT, 1, DEBUG_TEXT_COLOR, 2, cv2.LINE_AA)
            cv2.circle(debug_display, (target_screen_x, target_screen_y), DEBUG_CIRCLE_RADIUS, DEBUG_CIRCLE_COLOR, -1)
            cv2.imshow('Debug Display', debug_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        for _ in range(DEBUG_SAMPLES_PER_POINT):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            current_eye_x_norm, current_eye_y_norm = None, None
            mapped_mouse_x, mapped_mouse_y = None, None
            actual_mouse_x, actual_mouse_y = pyautogui.position()
            pitch, yaw, roll = None, None, None

            # Create a black debug display for this frame
            debug_display = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            # Draw the target yellow circle
            cv2.circle(debug_display, (target_screen_x, target_screen_y), DEBUG_CIRCLE_RADIUS, DEBUG_CIRCLE_COLOR, -1)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get eye center
                    eye_center = get_eye_center(face_landmarks.landmark)
                    if eye_center:
                        current_eye_x_norm, current_eye_y_norm = eye_center[0], eye_center[1]

                        # Map eye coordinates to screen coordinates
                        clamped_eye_x = np.clip(current_eye_x_norm, calibration_data['eye_x_min'], calibration_data['eye_x_max'])
                        clamped_eye_y = np.clip(current_eye_y_norm, calibration_data['eye_y_min'], calibration_data['eye_y_max'])

                        mapped_mouse_x = int(np.interp(clamped_eye_x,
                                                       [calibration_data['eye_x_min'], calibration_data['eye_x_max']],
                                                       [0, screen_width]))
                        mapped_mouse_y = int(np.interp(clamped_eye_y,
                                                       [calibration_data['eye_y_min'], calibration_data['eye_y_max']],
                                                       [0, screen_height]))

                        # Draw gaze point on webcam feed
                        frame_h, frame_w, _ = frame.shape
                        gaze_x_on_frame = int(current_eye_x_norm * frame_w)
                        gaze_y_on_frame = int(current_eye_y_norm * frame_h)
                        cv2.circle(frame, (gaze_x_on_frame, gaze_y_on_frame), 5, (0, 255, 0), -1) # Green dot

                        # Draw the actual mapped eye position as a green circle on the debug display
                        if mapped_mouse_x is not None and mapped_mouse_y is not None:
                            cv2.circle(debug_display, (mapped_mouse_x, mapped_mouse_y), DEBUG_CIRCLE_RADIUS // 2, (0, 255, 0), -1) # Green circle for actual
                            
                    # Get head pose
                    pitch, yaw, roll = get_head_pose(face_landmarks.landmark, frame.shape)

            # Print debug info to console
            print(f"  Eye (norm): ({current_eye_x_norm:.4f}, {current_eye_y_norm:.4f}) "
                  f"Mapped Mouse: ({mapped_mouse_x}, {mapped_mouse_y}) "
                  f"Actual Mouse: ({actual_mouse_x}, {actual_mouse_y}) ", end='\r')
            if pitch is not None:
                print(f"Head Pose (P/Y/R): ({pitch:.2f}, {yaw:.2f}, {roll:.2f}) ", end='\r')
            else:
                print("Head Pose: N/A ", end='\r')

            # Display webcam feed
            cv2.imshow('Eye Tracker Debug', frame)

            # Display debug display with target and actual mapped point
            cv2.imshow('Debug Display', debug_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("\n" + "="*80) # New line after each point's samples

    cap.release()
    cv2.destroyAllWindows()
    print("Debug mode stopped.")

if __name__ == "__main__":
    run_debug_mode()
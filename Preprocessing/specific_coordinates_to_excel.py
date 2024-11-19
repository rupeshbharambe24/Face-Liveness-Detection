import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Define action-specific landmarks
ACTION_LANDMARKS = {
    "1smile": [13, 45],
    "9open-mouth": [26, 30, 36, 45, 46, 49, 50],
    "6eye-close": [10, 11, 13, 21],
}

# Dataset folder path
dataset_path = "R:\Projects\Face Liveliness Detection\Codes\dataset"
output_data = []

# Process each subfolder and images
for action_folder in os.listdir(dataset_path):
    action_path = os.path.join(dataset_path, action_folder)
    if os.path.isdir(action_path) and action_folder in ACTION_LANDMARKS:
        relevant_landmarks = ACTION_LANDMARKS[action_folder]  # Get landmarks for the action
        for image_file in os.listdir(action_path):
            image_path = os.path.join(action_path, image_file)
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image with FaceMesh
                results = face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = []
                        for idx in relevant_landmarks:
                            lm = face_landmarks.landmark[idx]
                            landmarks.extend([lm.x, lm.y, lm.z])
                        # Append data with action label
                        output_data.append([action_folder, image_file] + landmarks)

# Dynamically adjust the columns based on the data
if output_data:
    # Determine the maximum number of columns dynamically
    max_landmarks = max(len(row) for row in output_data) - 2  # Exclude 'Action' and 'Image' columns
    columns = ["Action", "Image"] + [f"Landmark_{i}_x" if i % 3 == 0 else (f"Landmark_{i // 3}_y" if i % 3 == 1 else f"Landmark_{i // 3}_z") for i in range(max_landmarks)]

    # Normalize all rows to have the same number of elements (fill missing with None)
    for row in output_data:
        while len(row) < len(columns):
            row.append(None)

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(output_data, columns=columns)
    df.to_excel("action_specific_landmarks.xlsx", index=False)
    print("Action-specific landmarks saved to action_specific_landmarks.xlsx")
else:
    print("No face landmarks detected in the dataset.")

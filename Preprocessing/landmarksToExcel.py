import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Dataset folder path
dataset_path = r"R:\Projects\Face Liveliness Detection\Codes\facial_action_final"
output_data = []

# Process each subfolder and images
for pose_folder in os.listdir(dataset_path):
    pose_path = os.path.join(dataset_path, pose_folder)
    if os.path.isdir(pose_path):
        pose_label = pose_folder  # e.g., "pose1"
        for image_file in os.listdir(pose_path):
            image_path = os.path.join(pose_path, image_file)
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process the image with FaceMesh
                results = face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks_xy = []
                        for i, landmark in enumerate(face_landmarks.landmark):
                            landmarks_xy.extend([landmark.x, landmark.y])
                        # Append data with pose label, image file, and landmark coordinates
                        output_data.append([pose_label, image_file] + landmarks_xy)

# Dynamically adjust the columns based on the first row of data
if output_data:
    num_landmarks = len(output_data[0]) - 2  # Exclude 'Pose' and 'Image'
    columns = ["Pose", "Image"] + [f"Landmark_{i}_x" for i in range(1, num_landmarks // 2 + 1)] + [f"Landmark_{i}_y" for i in range(1, num_landmarks // 2 + 1)]

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(output_data, columns=columns)
    df.to_excel("face_landmarks_xy.xlsx", index=False)
    print("Face landmarks (x, y) saved to face_landmarks_xy.xlsx")
else:
    print("No face landmarks detected in the dataset.")
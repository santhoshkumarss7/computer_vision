import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load image from local file system
image_path = r'C:\Users\DELL\Desktop\github_cv\pose_estimation\pose_input.jpg'  # Replace with your image file path
image = cv2.imread(image_path)

# Process the image
results = mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Check if pose landmarks are detected
if results.pose_landmarks:
    # Draw landmarks and connections on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # Display the annotated image using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Pose Estimation")
    plt.axis('off')  # Hide axis ticks and labels
    plt.show()
else:
    print("No pose landmarks detected in the image.")

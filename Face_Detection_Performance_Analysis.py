# Install necessary packages
!pip install retina-face
!pip install opencv-python retinaface dlib mtcnn torch torchvision
!pip install mtcnn
!pip install pillow opencv-python dlib retinaface mtcnn torch matplotlib tqdm pandas numpy
!pip install ultralytics

# Import necessary libraries
import os
import time
import bz2
import urllib.request
import cv2
import dlib
from retinaface import RetinaFace
from mtcnn import MTCNN
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance  # For image augmentation
from ultralytics import YOLO

# The below lines ensure that DLIB's CNN face detector model is downloaded if not already present.
# DLIB requires a pre-trained model for its CNN-based face detector.
if not os.path.isfile('mmod_human_face_detector.dat'):
    print("Downloading DLIB's CNN face detector model...")
    url = 'http://dlib.net/files/mmod_human_face_detector.dat.bz2'
    urllib.request.urlretrieve(url, 'mmod_human_face_detector.dat.bz2')
    # Decompress the bz2 file
    with bz2.open('mmod_human_face_detector.dat.bz2', 'rb') as f_in:
        with open('mmod_human_face_detector.dat', 'wb') as f_out:
            f_out.write(f_in.read())
    print("Download complete.")

# Load face detection models:
# 1. Haar Cascade (classical method based on Haar features)
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. HOG-based face detector (Dlib)
hog_face_detector = dlib.get_frontal_face_detector()

# 3. CNN-based face detector (Dlib's CNN model)
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

# 4. MTCNN (uses a multi-stage CNN approach)
mtcnn_detector = MTCNN()

# 5. YOLO model (Replace "/content/best.pt" with your YOLO model path)
# This can be a custom trained YOLO model for face detection.
yolo_model = YOLO("/content/best.pt")

# Lists to store performance metrics for images and videos
performance_data_images = []
performance_data_videos = []

def augment_image(image):
    """
    Applies random augmentation to an image:
    - Random rotation
    - Random scaling
    - Random flipping
    - Random brightness adjustment
    
    This is useful to test how the detection algorithms perform on slightly
    modified/augmented versions of the images.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Random rotation
    if np.random.rand() > 0.5:
        angle = np.random.randint(-30, 30)
        pil_image = pil_image.rotate(angle)

    # Random scaling
    if np.random.rand() > 0.5:
        scale_factor = np.random.uniform(0.8, 1.2)
        new_width = int(pil_image.width * scale_factor)
        new_height = int(pil_image.height * scale_factor)
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Random flipping (horizontal flip)
    if np.random.rand() > 0.5:
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness adjustment
    if np.random.rand() > 0.5:
        enhancer = ImageEnhance.Brightness(pil_image)
        brightness_factor = np.random.uniform(0.5, 1.5)
        pil_image = enhancer.enhance(brightness_factor)

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def retina_detect_faces_in_images(dataset_path, output_folder):
    """
    Processes a folder of images using RetinaFace.
    For each image:
    - Reads the image
    - Optionally applies augmentation
    - Uses RetinaFace to detect faces
    - Draws bounding boxes around detected faces
    - Saves processed image
    - Records performance metrics (processing time, number of faces detected)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in tqdm(os.listdir(dataset_path), desc="RetinaFace - Images"):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dataset_path, image_file)
            output_path = os.path.join(output_folder, image_file)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_path}")
                continue

            # Apply augmentation before detection if desired
            image = augment_image(image)

            start_time = time.time()
            # Detect faces using RetinaFace
            faces = RetinaFace.detect_faces(image)
            detection_time = time.time() - start_time

            num_faces_detected = len(faces)

            # Draw bounding boxes around detected faces
            for face in faces.values():
                facial_area = face['facial_area']
                cv2.rectangle(image, (facial_area[0], facial_area[1]),
                              (facial_area[2], facial_area[3]), (255, 0, 0), 2)

            cv2.imwrite(output_path, image)
            print(f"Processed image saved as {output_path}")

            # Append performance data
            performance_data_images.append({
                'Algorithm': 'RetinaFace',
                'Image': image_file,
                'Faces Detected': num_faces_detected,
                'Processing Time': detection_time
            })

def detect_faces_in_images(method, dataset_path, output_folder, rectangle_color=(255, 0, 0)):
    """
    Processes a folder of images with a specified detection method (haar, hog, cnn, mtcnn, yolo).
    For each image:
    - Reads and augments the image
    - Detects faces using the chosen method
    - Draws bounding boxes
    - Saves processed image
    - Records performance data
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in tqdm(os.listdir(dataset_path), desc=f"{method.upper()} - Images"):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dataset_path, image_file)
            output_path = os.path.join(output_folder, image_file)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image: {image_path}")
                continue

            # Optional: Augment the image before detection
            image = augment_image(image)

            start_time = time.time()
            num_faces_detected = 0

            # Haar Cascade detection
            if method == 'haar':
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1,
                                                      minNeighbors=5, minSize=(30, 30))
                num_faces_detected = len(faces)
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_color, 2)

            # HOG-based detection (Dlib)
            elif method == 'hog':
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = hog_face_detector(gray_image, 1)
                num_faces_detected = len(faces)
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
                    cv2.rectangle(image, (x, y), (w, h), rectangle_color, 2)

            # CNN-based detection (Dlib)
            elif method == 'cnn':
                faces = cnn_face_detector(image, 1)
                num_faces_detected = len(faces)
                for face in faces:
                    x, y, w, h = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
                    cv2.rectangle(image, (x, y), (w, h), rectangle_color, 2)

            # MTCNN
            elif method == 'mtcnn':
                faces = mtcnn_detector.detect_faces(image)
                num_faces_detected = len(faces)
                for face in faces:
                    x, y, w, h = face['box']
                    cv2.rectangle(image, (x, y), (x + w, y + h), rectangle_color, 2)

            # YOLO
            elif method == 'yolo':
                results = yolo_model(image)
                boxes = results[0].boxes
                for box in boxes:
                    top_left_x = int(box.xyxy[0][0])
                    top_left_y = int(box.xyxy[0][1])
                    bottom_right_x = int(box.xyxy[0][2])
                    bottom_right_y = int(box.xyxy[0][3])
                    confidence = box.conf[0]
                    num_faces_detected += 1
                    cv2.rectangle(image, (top_left_x, top_left_y),
                                  (bottom_right_x, bottom_right_y), rectangle_color, 2)
                    label = f"Confidence: {confidence:.2f}"
                    cv2.putText(image, label, (top_left_x, top_left_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 1)

            detection_time = time.time() - start_time
            cv2.imwrite(output_path, image)
            print(f"Processed image saved as {output_path}")

            performance_data_images.append({
                'Algorithm': method.upper(),
                'Image': image_file,
                'Faces Detected': num_faces_detected,
                'Processing Time': detection_time
            })

def retina_detect_faces_in_video(video_path, output_video_path, video_file):
    """
    Processes a single video using RetinaFace:
    - Reads frames from the video
    - Detects faces using RetinaFace in each frame
    - Draws bounding boxes
    - Writes processed frames into a new output video
    - Records performance metrics (frame-wise)
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Default to 30 if unable to read FPS

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        faces = RetinaFace.detect_faces(frame)
        detection_time = time.time() - start_time

        num_faces_detected = len(faces)

        for face in faces.values():
            facial_area = face['facial_area']
            cv2.rectangle(frame, (facial_area[0], facial_area[1]),
                          (facial_area[2], facial_area[3]), (255, 0, 0), 2)

        out.write(frame)

        performance_data_videos.append({
            'Algorithm': 'RetinaFace',
            'Video': video_file,
            'Frame': frame_number,
            'Faces Detected': num_faces_detected,
            'Processing Time': detection_time
        })

        frame_number += 1

    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")

def detect_faces_in_video(method, video_path, output_video_path, video_file, rectangle_color=(255, 0, 0)):
    """
    Processes a single video using the specified detection method:
    - Reads frames
    - Detects faces (haar, hog, cnn, mtcnn, or yolo)
    - Draws bounding boxes
    - Writes processed frames out
    - Records performance metrics for each frame
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        num_faces_detected = 0

        # Haar
        if method == 'haar':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(30, 30))
            num_faces_detected = len(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

        # HOG
        elif method == 'hog':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = hog_face_detector(gray_frame, 1)
            num_faces_detected = len(faces)
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(frame, (x, y), (w, h), rectangle_color, 2)

        # CNN
        elif method == 'cnn':
            faces = cnn_face_detector(frame, 1)
            num_faces_detected = len(faces)
            for face in faces:
                x, y, w, h = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
                cv2.rectangle(frame, (x, y), (w, h), rectangle_color, 2)

        # MTCNN
        elif method == 'mtcnn':
            faces = mtcnn_detector.detect_faces(frame)
            num_faces_detected = len(faces)
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

        # YOLO
        elif method == 'yolo':
            results = yolo_model(frame)
            boxes = results[0].boxes
            for box in boxes:
                top_left_x = int(box.xyxy[0][0])
                top_left_y = int(box.xyxy[0][1])
                bottom_right_x = int(box.xyxy[0][2])
                bottom_right_y = int(box.xyxy[0][3])
                confidence = box.conf[0]
                num_faces_detected += 1

                cv2.rectangle(frame, (top_left_x, top_left_y),
                              (bottom_right_x, bottom_right_y), rectangle_color, 2)
                label = f"Confidence: {confidence:.2f}"
                cv2.putText(frame, label, (top_left_x, top_left_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 1)

        detection_time = time.time() - start_time

        out.write(frame)

        performance_data_videos.append({
            'Algorithm': method.upper(),
            'Video': video_file,
            'Frame': frame_number,
            'Faces Detected': num_faces_detected,
            'Processing Time': detection_time
        })

        frame_number += 1

    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")

# Paths to datasets (adjust as per your setup)
image_dataset_path = '/content/Images/'  # Folder with images
video_dataset_path = '/content/Videos/'  # Folder with videos
output_base_path = '/content/Outputs/'   # Base folder for outputs

# Process images using RetinaFace
retina_images_output_folder = os.path.join(output_base_path, 'retinaface_images')
retina_detect_faces_in_images(image_dataset_path, retina_images_output_folder)

# Process images using other methods (Haar, HOG, CNN, MTCNN, YOLO)
methods = ['haar', 'hog', 'cnn', 'mtcnn', 'yolo']
for method in methods:
    method_images_output_folder = os.path.join(output_base_path, f'{method}_images')
    detect_faces_in_images(method, image_dataset_path, method_images_output_folder)

# Process videos using RetinaFace
retina_videos_output_folder = os.path.join(output_base_path, 'retinaface_videos')
os.makedirs(retina_videos_output_folder, exist_ok=True)
video_files = [file for file in os.listdir(video_dataset_path) if file.endswith(('.mp4', '.avi'))]
for video_file in tqdm(video_files, desc="RetinaFace - Videos"):
    video_path = os.path.join(video_dataset_path, video_file)
    output_video_path = os.path.join(retina_videos_output_folder, video_file)
    retina_detect_faces_in_video(video_path, output_video_path, video_file)

# Process videos using other methods
for video_file in video_files:
    video_path = os.path.join(video_dataset_path, video_file)
    for method in methods:
        method_videos_output_folder = os.path.join(output_base_path, f'{method}_videos')
        os.makedirs(method_videos_output_folder, exist_ok=True)
        output_video_path = os.path.join(method_videos_output_folder, video_file)
        detect_faces_in_video(method, video_path, output_video_path, video_file)

# Save the performance data to CSV files for later analysis
performance_df_images = pd.DataFrame(performance_data_images)
performance_df_videos = pd.DataFrame(performance_data_videos)

performance_df_images.to_csv(os.path.join(output_base_path, 'performance_images.csv'), index=False)
performance_df_videos.to_csv(os.path.join(output_base_path, 'performance_videos.csv'), index=False)

print("Performance data saved.")

def analyze_and_rank_algorithms(performance_df, data_type='Images', output_path='plots'):
    """
    Analyzes and ranks algorithms based on their performance data.
    For each algorithm, we compute:
    - Mean processing time
    - Total faces detected (sum across all samples)
    
    Then we rank algorithms based on:
    - Detection rank (higher faces detected = better)
    - Speed rank (lower processing time = better)
    
    The overall rank is the average of detection rank and speed rank.
    
    Finally, we plot bar charts for these metrics and save them.
    """
    print(f"\nAnalysis and Ranking for {data_type}:")

    # Create output directory for plots if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Group the data by Algorithm
    grouped = performance_df.groupby('Algorithm')

    # Compute metrics
    metrics = grouped.agg({
        'Processing Time': 'mean',
        'Faces Detected': 'sum'
    }).reset_index()

    # Ensure Faces Detected is int
    metrics['Faces Detected'] = metrics['Faces Detected'].astype(int)

    # Rank algorithms
    # Detection Rank: rank by number of faces (more is better, so ascending=False)
    metrics['Detection Rank'] = metrics['Faces Detected'].rank(method='min', ascending=False)
    # Speed Rank: rank by processing time (less is better, so ascending=True)
    metrics['Speed Rank'] = metrics['Processing Time'].rank(method='min')
    # Overall rank: average of detection and speed ranks
    metrics['Overall Rank'] = (metrics['Detection Rank'] + metrics['Speed Rank']) / 2
    metrics = metrics.sort_values('Overall Rank')

    # Print the metrics for inspection
    print(metrics[['Algorithm', 'Processing Time', 'Faces Detected', 'Detection Rank', 'Speed Rank', 'Overall Rank']])

    # Create and save plots
    plt.figure(figsize=(20, 4))

    metrics_to_plot = [
        ('Processing Time', 'Processing Time (seconds)', 'orange'),
        ('Faces Detected', 'Total Faces Detected', 'green'),
        ('Detection Rank', 'Detection Rank', 'blue'),
        ('Speed Rank', 'Speed Rank', 'purple'),
        ('Overall Rank', 'Overall Rank', 'red')
    ]

    # Plot each metric as a bar chart
    for i, (col, ylabel, color) in enumerate(metrics_to_plot, start=1):
        plt.subplot(1, 5, i)
        plt.bar(metrics['Algorithm'], metrics[col], color=color)
        plt.title(f'{data_type} - {col}')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{data_type.lower()}_metrics.png'))
    plt.close()

    return metrics

def main(output_base_path='output'):
    """
    Main function to:
    - Analyze and rank algorithms for images and videos separately
    - Combine results for overall ranking
    - Generate plots for images, videos, and combined
    """
    # Create output directories for plots
    plots_path = os.path.join(output_base_path, 'plots')
    os.makedirs(plots_path, exist_ok=True)

    # Analyze and rank algorithms for images
    metrics_images = analyze_and_rank_algorithms(performance_df_images,
                                                 data_type='Images',
                                                 output_path=plots_path)

    # Analyze and rank algorithms for videos
    metrics_videos = analyze_and_rank_algorithms(performance_df_videos,
                                                 data_type='Videos',
                                                 output_path=plots_path)

    # Combine image and video metrics to determine overall performance
    combined_metrics = pd.concat([metrics_images, metrics_videos])

    overall_metrics = combined_metrics.groupby('Algorithm').agg({
        'Processing Time': 'mean',
        'Faces Detected': 'sum'
    }).reset_index()

    overall_metrics['Faces Detected'] = overall_metrics['Faces Detected'].astype(int)
    overall_metrics['Detection Rank'] = overall_metrics['Faces Detected'].rank(method='min', ascending=False)
    overall_metrics['Speed Rank'] = overall_metrics['Processing Time'].rank(method='min')
    overall_metrics['Overall Rank'] = (overall_metrics['Detection Rank'] + overall_metrics['Speed Rank']) / 2
    overall_metrics = overall_metrics.sort_values('Overall Rank')

    print("\nOverall Ranking Across Images and Videos:")
    print(overall_metrics[['Algorithm', 'Processing Time', 'Faces Detected', 'Detection Rank', 'Speed Rank', 'Overall Rank']])

    # Plot overall metrics
    plt.figure(figsize=(20, 4))
    metrics_to_plot = [
        ('Processing Time', 'Processing Time (seconds)', 'orange'),
        ('Faces Detected', 'Total Faces Detected', 'green'),
        ('Detection Rank', 'Detection Rank', 'blue'),
        ('Speed Rank', 'Speed Rank', 'purple'),
        ('Overall Rank', 'Overall Rank', 'red')
    ]

    for i, (col, ylabel, color) in enumerate(metrics_to_plot, start=1):
        plt.subplot(1, 5, i)
        plt.bar(overall_metrics['Algorithm'], overall_metrics[col], color=color)
        plt.title(f'Overall - {col}')
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, 'overall_metrics.png'))
    plt.close()

    # Confirm and print the directory structure of processed outputs
    processed_structure = os.listdir(output_base_path)
    print("\nProcessed output structure:", processed_structure)

if __name__ == "__main__":
    main()

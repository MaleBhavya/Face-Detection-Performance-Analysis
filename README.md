Face Detection Performance Analysis


This project implements and evaluates multiple face detection algorithms on both images and videos. The following algorithms are included:

1) Haar Cascade

2) HOG-based Dlib Detector

3) CNN-based Dlib Detector

4) MTCNN

5) RetinaFace

6) YOLO


Features

1) Image Face Detection: Processes images using the specified detection algorithms and outputs results with bounding boxes.

2) Video Face Detection: Processes video frames to detect faces, saving the output with bounding boxes drawn.

3) Performance Analysis: Logs detection times, the number of faces detected, and other metrics for comparative analysis.

4) Image Augmentation: Random augmentations applied to test algorithm robustness.


Requirements

Install the following Python packages:

pip install retina-face opencv-python dlib mtcnn torch torchvision ultralytics pillow matplotlib tqdm pandas numpy


Dataset Structure

Organize your dataset as follows:

/content/
    Images/
        image1.jpg
        image2.png
        ...
    Videos/
        video1.mp4
        video2.avi
        ...
    Outputs/
        (Generated results and performance data will be saved here)


Usage

Clone this repository:

git clone <repository_url>
cd <repository_folder>

Ensure the datasets are placed in the appropriate folders (see "Dataset Structure").

Run the script:

python main.py

Processed outputs will be saved in the Outputs/ folder.


Outputs

1) Processed Images: Saved with bounding boxes drawn for detected faces.

2) Processed Videos: Saved with bounding boxes drawn frame by frame.

3) Performance Data: CSV files with metrics such as processing time and faces detected:

performance_images.csv

performance_videos.csv


Algorithms

1) Haar Cascade

Classical method based on Haar features.

2) HOG-based Dlib Detector

Histogram of Oriented Gradients for feature detection.

3) CNN-based Dlib Detector

CNN model from Dlib for face detection.

4) MTCNN

Multi-task Cascaded Convolutional Networks for face detection.

5) RetinaFace

State-of-the-art face detection algorithm.

6) YOLO

Custom-trained YOLO model for face detection.


Results

Use the generated performance CSV files for analysis. You can visualize metrics using tools such as Pandas and Matplotlib.






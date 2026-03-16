# 1. 📌 Real-Time Face Detection using MediaPipe & Opencv

A simple and efficient real-time face detection system built using MediaPipe and OpenCV.
This project detects faces from a live webcam feed and displays bounding boxes with confidence scores and FPS

## 🛠️ Technologies Used: 

     Python 3

     OpenCV

     MediaPipe

     NumPy


## Face Detection Initialization:

     mpFaceDetection = mp.solutions.face_detection
     faceDetection = mpFaceDetection.FaceDetection()


## 🧠 How It Works:

   🔹Webcam captures real-time video.

   🔹Frames are converted from BGR to RGB (required by MediaPipe).

   🔹MediaPipe Face Detection model processes the image.

 ## For each detected face:
 
     Bounding box is calculated

     Confidence score is displayed

     FPS is calculated and shown on screen.

## ⚙️ Install dependencies:
            
      pip install opencv-python mediapipe numpy

---

# 2. 📌 Real Time face blur using opencv:

This project detects human faces in real-time using OpenCV’s Haar Cascade classifier and applies a Gaussian blur effect to anonymize them.

   It captures video from the webcam, detects faces frame-by-frame, and blurs only the detected face regions while keeping the rest of the frame unchanged.

   
## 🛠️ Technologies used:

        Python 3

        OpenCV

        Haar Cascade Classifier (haarcascade_frontalface_default.xml)

## 🧠 How it works:


🔹Capture webcam video using cv2.VideoCapture()

🔹Convert frame to grayscale

🔹Detect faces using detectMultiScale()

🔹Extract Region of Interest (ROI)

🔹Apply cv2.GaussianBlur() to the face

🔹Replace blurred ROI back into original frame

🔹Display processed frame

## 📸 Output:

![Face blur Output](output/faceblur.png)

---

# 3. 📌 Real time face detection using Opencv's Haarcascade

   This project implements a real-time face detection system using OpenCV’s pre-trained Haar Cascade classifier. The system captures live video from the webcam, detects human faces in each frame, and draws bounding boxes around them.


## 🛠️ Technologies used:

      Python
      OpenCV
      Haar Cascade Classifier

## 🧠 How it works:

   i.Load Haar Cascade Classifier
     The program loads OpenCV’s pre-trained haarcascade_frontalface_default.xml model for detecting frontal faces.

   ii.Initialize Webcam
      The webcam is accessed using cv2.VideoCapture() to capture live video frames.

   iii.Frame Processing
       Each frame is converted to grayscale.

   iv.The Haar Cascade classifier detects faces in the grayscale image.
       For every detected face, a blue rectangle is drawn around it.



---

# 4. 📌 Real-Time Pencil Sketch Using OpenCV

This project converts live webcam video into a real-time pencil sketch effect using OpenCV.


## It displays:
  
   🔹Original camera feed

   🔹Pencil sketch version

   🔹Both shown side-by-side in one window

## 🛠️ Technologies used:
      
     Python3
     OpenCV
     NumPy

## 🧠 How it works:

  🔹Convert frame to grayscale

  🔹Invert grayscale image

  🔹Apply Gaussian Blur

  🔹Blend using Dodge technique


## 📸 Output

![livesketch Output](output/livesketch.png)


---
# 5. 📌 Real time facial landmark detection using dlib and opencv

   A real-time Facial Landmark Detection system built using Dlib and OpenCV.
   This project detects faces from webcam input and plots 68 facial landmark points (eyes, nose, mouth, jawline, etc.) on the    detected face.

## 🔮 Features:


   🔹Real-time webcam face detection

   🔹68 facial landmark detection

   🔹Landmark visualization with OpenCV

   🔹Fast and lightweight implementation

   🔹Clean and simple code structure

## 🛠️ Technologies used:
     
      Python 3.x
      OpenCV
      Dlib
      NumPy

## Model Used:

This project uses:

     Dlib's Frontal Face Detector
     shape_predictor_68_face_landmarks.dat model (pre-trained)

## ⚙️  Install dependencies:
      
       pip install opencv-python
       pip install dlib
       pip install numpy

##  📸 Output:

 ![face_landmark Output](output/facelandmark.png)

---

 # 6. 📌 Real-Time Head Pose Estimation using OpenCV & dlib

## 📖 Overview

This project implements **real-time head pose estimation** using a webcam.
It detects a face, extracts facial landmarks, and estimates the head orientation (pose) using the **solvePnP** algorithm from OpenCV.

The output is visualized as a **line projecting from the nose**, indicating the head direction.



## 🔮 Features

   🔹Real-time webcam head pose estimation
   
   🔹Facial landmark detection using dlib (68-point model)
   
   🔹Pose estimation using OpenCV `solvePnP`
   
   🔹Visualization of head direction with nose projection
   
   🔹Works for single or multiple faces



## 🧠 Working Pipeline

```
Webcam Frame
     ↓
Face Detection (dlib HOG detector)
     ↓
Facial Landmark Detection (68 points)
     ↓
Select key landmarks (nose, chin, eyes, mouth)
     ↓
solvePnP → Rotation & Translation vectors
     ↓
projectPoints → Nose direction
     ↓
Display head pose visualization
```



## 🛠️ Requirements

⚙️ Install dependencies:

```bash
pip install opencv-python dlib numpy
```

Download the pretrained landmark model:
   **shape_predictor_68_face_landmarks.dat**

Place it in the project folder.




##  Key Concepts Used

### 🔹 Facial Landmark Detection

dlib detects 68 facial keypoints.
This project uses:

* Nose tip → 30
* Chin → 8
* Left eye corner → 36
* Right eye corner → 45
* Left mouth → 48
* Right mouth → 54



### 🔹 3D Model Points

Predefined approximate 3D face model used for pose estimation.



### 🔹 solvePnP

Computes:

* Rotation vector → Head orientation
* Translation vector → Face position



### 🔹 projectPoints

Projects a 3D point onto the image plane to draw a direction line from the nose.



## 📸 Output

![head pose Output](output/headpose.png)

---

# 7. 😴 Driver Drowsiness Detection System

## 📌 Overview

The Driver Drowsiness Detection System is a computer vision–based safety application that monitors a person's eye activity using a webcam to detect signs of drowsiness. When the system detects that the eyes remain closed beyond a defined threshold, it triggers both visual and audio alerts to wake the person.

This system is designed to help reduce accidents caused by driver fatigue by providing real-time monitoring and warning mechanisms.

## 🎯 Objective

The goal of this project is to build a real-time system that:

  Detects a human face using computer vision

  Tracks facial landmarks to locate the eyes

  Calculates the Eye Aspect Ratio (EAR) to determine eye closure

  Triggers alerts when drowsiness is detected

## ⚙️ How It Works: 

```
Webcam Video Input
        │
        ▼
Face Detection (dlib)
        │
        ▼
Facial Landmark Detection
        │
        ▼
Eye Landmark Extraction
        │
        ▼
Eye Aspect Ratio (EAR) Calculation
        │
        ▼
Check Threshold Value
        │
        ▼
If EAR < Threshold → Drowsiness Detected
        │
        ▼
Visual Alert + Voice Warning
```

## Process Explanation

🔹The webcam captures real-time video frames.

🔹The system detects faces using dlib’s frontal face detector.

🔹Facial landmarks are extracted using the 68-point landmark predictor.

🔹Eye coordinates are identified from the facial landmarks.

🔹The Eye Aspect Ratio (EAR) is calculated using Euclidean distance.

🔹If the EAR falls below the threshold (0.25), the system assumes the eyes are closed.

🔹A visual warning message and voice alert are triggered.

## 🛠️ Technologies Used

    Python
    OpenCV – real-time computer vision processing
    dlib – face detection and facial landmark detection
    SciPy – Euclidean distance calculations
    pyttsx3 – text-to-speech alert system

## ✨ Features:

🔹Real-time face detection

🔹Eye tracking using facial landmarks

🔹Eye Aspect Ratio (EAR) based drowsiness detection

🔹Visual alert messages

🔹Audio warning system

🔹Works with a standard webcam

## 📦 Installation

1️⃣ Install Required Libraries
     
    pip install opencv-python dlib scipy pyttsx3
2️⃣ Download Facial Landmark Model

Download the dlib facial landmark predictor file:

    shape_predictor_68_face_landmarks.dat

Place it in the same project directory.

 download it from:

    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2


   The webcam will start and the system will monitor eye activity in real time.

## 📸 Example Output

    When drowsiness is detected:

    DROWSINESS DETECTED
    Alert!!!! WAKE UP DUDE

An audio warning will also be played using text-to-speech.

⚠️ Threshold Value

The system uses an Eye Aspect Ratio threshold of 0.25.

    EAR < 0.25 → Eyes Closed → Drowsiness Alert

This value can be adjusted depending on user behavior and environment.


## 🌍 Applications

- Driver safety monitoring systems

- Industrial worker fatigue detection

- Research in human behavior monitoring

- Healthcare fatigue monitoring

# 8.👤 Real-Time Age and Gender Detection using YOLOv8 and DeepFace

## 📌 Overview

This project implements a **real-time age and gender detection system** using computer vision and deep learning. The system detects faces from a webcam feed using **YOLOv8 face detection**, then analyzes each detected face to estimate **age and gender** using **DeepFace**.

The detected information is displayed on the screen along with bounding boxes around the detected faces.

This project demonstrates how **object detection and facial analysis models** can be combined to build intelligent real-time applications.

---

# 🎯 Objective

The objective of this project is to create a system that can:

- Detect human faces in real time
- Predict the **age** of the detected person
- Identify the **gender**
- Display results directly on the video stream
- Run efficiently on a webcam feed

---

# ⚙️ How It Works

```
Webcam Video Input
        │
        ▼
Face Detection (YOLOv8)
        │
        ▼
Face Cropping
        │
        ▼
Facial Analysis (DeepFace)
        │
        ▼
Age Prediction
        │
        ▼
Gender Prediction
        │
        ▼
Display Results on Video Frame
```

---

# 🛠️ Technologies Used

- **Python**
- **OpenCV** – Video processing and visualization
- **Ultralytics YOLOv8** – Face detection
- **DeepFace** – Age and gender analysis
- **NumPy** – Array processing

---

# ✨ Features

- Real-time face detection
- Age estimation from facial features
- Gender prediction
- Live webcam processing
- Bounding box visualization
- Fast inference using YOLOv8

---

# 📦 Installation

## 1️⃣ Install Required Libraries

```bash
pip install opencv-python ultralytics deepface
```

---

## 2️⃣ Download YOLOv8 Face Model

Download the face detection model:

```
yolov8n-face.pt
```

Place the model file in the same directory as the Python script.


---

# 📸 Output

The system will display:

- A bounding box around detected faces
- Predicted **age**
- Predicted **gender**

Example label:

```
Male, 25
Female, 32
```

Displayed above the detected face in the video frame.

---

# 🚀 Future Enhancements

- Add **emotion detection**
- Support **multiple faces simultaneously**
- Improve accuracy using larger YOLO models
- Deploy as a **web application**
- Integrate **face recognition for identity detection**
- Add **real-time analytics dashboard**

---

# 🌍 Applications

- Smart surveillance systems
- Retail customer analytics
- Interactive marketing systems
- Human-computer interaction
- Demographic analysis

---







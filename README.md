# Driver-Drowsiness-Detection
A mini project that i completed in the 5th semester of my btech in computer science and technology from karunya university. This project was done using neural networks and could detect whether the user was tired or not by reading his/her eyes and face expressions using a live camera feed.

<br>  
<br>  
****
The increasing number of road accidents and life-threatening consequences is making life miserable on roads. In a fast-moving world, with so much of pollution, commotion and stress around driver fatigue is one of the major causes of accidents in the world. Drowsiness caused by immense fatigue and stress is a reason for the increased rate. Detecting the drowsiness of the driver is one of the surest ways of measuring driver fatigue.
The objective of this intermediate Python project is to build a prototype drowsiness detection system that will detect that a person’s eyes are closed for a few seconds and then alert the driver when drowsiness is detected.
The system so designed is a non-intrusive real-time monitoring system. The priority is on improving the safety of the driver without being obtrusive. In this project the eye blink of the driver is detected. If the driver’s eyes remain closed for more than a certain period of time, the driver is said to be drowsy and an alarm is sounded.
The programming for this is done in OpenCV using the Haarcascade library for the detection of facial features.
In this Python project, we will be using OpenCV for gathering the images from webcam and feed them into a Deep Learning model which will classify whether the person’s eyes are ‘Open’ or ‘Closed’.
The approach we will be using for this Python project is as follows
Step 1 – Take image as input from a camera.
Step 2 – Detect the face in the image and create a Region of Interest (ROI).
Step 3 – Detect the eyes from ROI and feed it to the classifier.
Step 4 – Classifier will categorize whether eyes are open or closed.
Step 5 – Calculate score to check whether the person is drowsy.

# Cat Detection and Door Opening System

This repository contains the code for a Cat Detection and Door Opening System. The system uses computer vision techniques to detect cat faces and authenticate them based on a set of registered cat images. Upon successful authentication, the system triggers the opening of a door, allowing the cat to pass through.

## Features

- **Firebase Authentication:** Allows users to register, sign in, and log out using Firebase Authentication.
- **Cat Face Detection:** Utilizes OpenCV and Haarcascades for detecting cat faces in video frames.
- **ORB Algorithm:** Implements the ORB (Oriented FAST and Rotated BRIEF) algorithm for feature extraction and matching during cat face recognition.
- **Structural Similarity Index (SSIM):** Measures the similarity between images to enhance cat face recognition accuracy.
- **Twilio Integration:** Sends SMS notifications to users when cat authentication fails, along with an image of the unrecognized cat.
- **Arduino Integration:** Controls the opening of a door by sending signals to an Arduino.

# Backend Setup
## Prerequisites 

Before running the system, ensure you have the following dependencies installed:

- Python
- OpenCV
- Flask
- Twilio
- Firebase
- Pyrebase
- NumPy
- RPi.GPIO (for Raspberry Pi GPIO control)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/cat-door-system.git
2. **Make a Virtual Environment by following**
   ```bash
    https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/
3. **Install the required Python packages**
    ```bash
   pip install -r requirements.txt
4. **Set up Firebase credentials:**
 - Create a Firebase project and obtain the configuration details.
 - Update the `firebase_config` dictionary in the code with your Firebase project details.
5. **Set up Twilio credentials:**
  - Obtain Twilio account SID, authentication token, and phone numbers.
  - Update the account_sid, auth_token, from_, and to variables in the code with your Twilio credentials.
6. **Set up Arduino:**
  - Connect the Arduino to the system and configure the GPIO pin in the code accordingly.
    
## Usage
   ```bash
   python app.py
   ```
  -  Navigate to http://localhost:5000

## Server Endpoint
- /register: Register a new user using Firebase Authentication.
- /signin: Sign in a user using Firebase Authentication.
- /logout: Log out a user using Firebase Authentication.
- /video_feed: Stream video feed with cat face detection.
- /open_door: Open the door manually.
- /register_cat: Stream video feed for cat registration.
- /getImagesBase64: Get a list of all JPEG images encoded as base64.
- /getImages/<image_name>: Get a specific image file.
- /getImages: Get a list of all JPEG images in the generated folder.


# Mobile Application Setup

This repository has linked the Mobile App Code as a sub-module, click on the submodule. 



## Prerequisites 

Before running the system, ensure that these critereas are met:

- Cat Door Server is running and the endpoints are accessible.
- Android Studios is installed on your system with a mobile emulator on the Device manager.
  

## Configuration

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/CatDoorMobile.git


3. **Installation of any packages is not neccessary since everything is configured in the AndroidManifest.xml and build.gradle files**
4. **Choose an android mobile emulator from Android Studio tools and run the application to get to the homepage**
5. **Inside the Java repository, we have all the activity files written in java**
6. **Inside of the Layout repository we have all the respective frontend .xml files for each activity**


## Home Page Activities
- Live Acitivty: Access the /video_feed endpoints and Shows the live feed of the cat door/ allows you to manually open the door.
- Upload Acitivty: access the /register endpoints and registers a new cat.
- Door Activity: Access the /open_door endpoints and allows the user to manually open the door. 
- Setting Activity: Allows the user to set up settings(under construction).
- Guide Activity: Shows a guide for how to use the application.
- Team Activity: Shows an About us page with the team who worked on the mobile application. 



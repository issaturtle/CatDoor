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

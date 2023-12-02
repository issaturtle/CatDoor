import cv2
import time
import threading
from skimage.metrics import structural_similarity as ssim
from flask import Flask, Response, request, jsonify, redirect, url_for
from twilio.rest import Client
import firebase_admin
from firebase_admin import credentials, auth
import keys
import os 
import serial
import pyrebase
import RPi.GPIO as GPIO
import time


app = Flask(__name__)

#Variables
firebase_config = {
    "apiKey": "AIzaSyAZx-aHNkVmADTd-ka_5qOxMKk5a45_CWI",
    "authDomain": "catdoor-b597c.firebaseapp.com",
    "projectId": "catdoor-b597c",
    "storageBucket": "catdoor-b597c.appspot.com",
    "messagingSenderId": "810484497038",
    "appId": "1:810484497038:web:d3f0892ac43715b7e967df",
    "measurementId": "G-3NW7F31SBZ",
    "databaseURL": ""
}

firebase = pyrebase.initialize_app(firebase_config)
auth2 = firebase.auth()
# Define a global variable switch and initialize it as False
switch = False
GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT) 


ser = ""
# Initialize the camera
camera = cv2.VideoCapture(0)
camera_lock = threading.Lock()
# Initialize the ORB detector
orb = cv2.ORB_create()
user = None
IMAGE_FOLDER = 'generated'

def send_signal_to_arduino():
    global camera_lock
    global switch
    global iteration
    
    try:
        
        print("hello from arduino")
        switch = True
        
        GPIO.output(26, GPIO.HIGH) 
        time.sleep(1)
        GPIO.output(26, GPIO.LOW) 
        time.sleep(5)

    except KeyboardInterrupt:
        GPIO.cleanup()

    
@app.route('/register', methods=['POST'])
def register():
    """
    Endpoint to register a new user using Firebase Authentication.
    Expects JSON data with 'email' and 'password' fields.
    Returns:
        JSON response containing the user's Firebase UID.
    """
    global user 
    try:
        request_data = request.get_json()
        email = request_data.get('email')
        password = request_data.get('password')

        # Create a new user account using Firebase Authentication
        user = auth2.create_user_with_email_and_password(email,password)

        # Return the user's Firebase UID
        return jsonify({"uid": user['localId']})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/signin', methods=['POST'])
def signin():
    """
    Endpoint to sign in a user using Firebase Authentication.
    Expects JSON data with 'email' and 'password' fields.
    Returns:
        JSON response containing the signed-in user's Firebase UID.
    """
    try:
        request_data = request.get_json()
        email = request_data.get('email')
        password = request_data.get('password')

        # Sign in the user using Firebase Authentication
        signed_in_user = auth2.sign_in_with_email_and_password(email, password)

        # Return the user's Firebase UID
        return jsonify({"uid": signed_in_user['localId']})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/logout', methods=['POST'])
def logout():
    """
    Endpoint to log out a user using Firebase Authentication.
    Returns:
        JSON response indicating successful user logout.
    """
    global user 
    if user is not None:
        # Use Firebase authentication method to sign out the user
        auth.logout()

        # Set the user variable to None
        user = None

    return jsonify({"message": "User logged out"})


# Continuously capture frames and process them
def capture_frames():
    """
    Continuously captures frames, processes them, and detects cat faces.
    Uses the ORB algorithm and SSIM for cat face recognition.
    Returns:
        Frame as a multipart response with JPEG content type.
    """
    global switch
    failed_attempts = 0
    max_failed_attempts = 20
    message_sent = 1
    first_time_delay = 25
    first_time_start = None 
    first_time_sent = True 
    directory_path = "generated"
    file_names = os.listdir(directory_path)
    # # Define the reference_cat_images as a global variable
    reference_cat_images = os.listdir(directory_path)
    for i in range(len(reference_cat_images)):
        reference_cat_images[i] =  directory_path + '/' + reference_cat_images[i]
    # reference_cat_images = ["cropped/1.jpg", "cropped/2.jpg", "cropped/3.jpg"]

    print("hello")
    while True:
        time.sleep(1)
        ret, frame = camera.read()
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the detected cat face
            captured_cat_face = gray[y:y + h, x:x + w]

            # Initialize an empty list to store SSIM scores
            ssim_scores = []

            # Initialize an empty list to store ORB keypoints and descriptors
            orb_keypoints = []
            orb_descriptors = []
            
            for reference_cat_image_path in reference_cat_images:
                # Load the reference cat image
                pic_of_cat = cv2.imread(reference_cat_image_path, cv2.IMREAD_GRAYSCALE)
                pic_of_cat = cv2.resize(pic_of_cat, (w, h))
                captured_cat_face = cv2.resize(captured_cat_face, pic_of_cat.shape[::-1])

                # Calculate SSIM between the two images
                similarity_index_value = ssim(pic_of_cat, captured_cat_face)

                # Store SSIM score
                ssim_scores.append(similarity_index_value)

                # Detect ORB keypoints and compute descriptors for the reference image
                reference_keypoints, reference_descriptor = orb.detectAndCompute(pic_of_cat, None)
                orb_keypoints.append(reference_keypoints)
                orb_descriptors.append(reference_descriptor)

            # Determine the dynamic threshold based on statistical analysis of SSIM scores
            if max(ssim_scores) > 0.4:
                # Detect ORB keypoints and compute descriptors for the captured cat face
                keypoints, captured_descriptor = orb.detectAndCompute(captured_cat_face, None)

                # Match ORB descriptors between the captured face and each reference image
                orb_matches = []
                for reference_descriptor in orb_descriptors:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(reference_descriptor, captured_descriptor)
                    matches = sorted(matches, key=lambda x: x.distance)
                    orb_matches.append(matches)

                # Compute a similarity score based on the number of good ORB matches and SSIM score
                max_orb_matches = max(len(matches) for matches in orb_matches)
                similarity_score = max_orb_matches + sum(ssim_scores)
                print(similarity_score)
                if similarity_score > 60:  # Adjust the threshold as needed
                    cv2.putText(frame, "Authenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    send_signal_to_arduino()
                    global switch 
                    if switch:
                        print("Switch is true")

                        print("Success")
                        switch = False
                    else:
                        switch = False
                        print("Switch is False")


                elif similarity_score <= 60:
                    cv2.putText(frame, "Incorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print(f"failed_attempts with ssim or orb: {failed_attempts}")

                    failed_attempts += 1
                    
                    if failed_attempts >= max_failed_attempts:
                        if first_time_sent == False:
                            # If it's the first time, start the delay timer
                            if first_time_start is None:
                                first_time_start = time.time()

                            # Check if the delay timer has reached 30 seconds
                            if time.time() - first_time_start >= first_time_delay:
                                first_time_sent = True  
                                first_time_start = None
                        else:
                            # Call a function to send the frame to the user using Twilio
                            send_frame_to_user(captured_cat_face)
                            first_time_sent = False 
                        failed_attempts = 0  # Reset failed attempts counter

            else:
                cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"failed_attempts without ssim or orb: {failed_attempts}")
                failed_attempts += 1
                if failed_attempts >= max_failed_attempts:
                    # Check if it's the first time to send the frame to the user
                    if first_time_sent == False:
                        # If it's the first time, start the delay timer
                        if first_time_start is None:
                            first_time_start = time.time()

                        # Check if the delay timer has reached 30 seconds
                        if time.time() - first_time_start >= first_time_delay:
                            first_time_sent = True  
                            first_time_start = None
                    else:
                        send_frame_to_user(captured_cat_face)
                        first_time_sent = False 
                    failed_attempts = 0  # Reset failed attempts counter
            
            # Draw a rectangle around the detected cat face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_bytes = encoded_frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
@app.route('/')
def index():
    return "Welcome to the cat detection and door opening system!"

@app.route('/video_feed')
def video_feed():
    """
    Endpoint to stream video feed with cat face detection.
    Returns:
        Multipart response with video frames.
    """    
    global camera

    if camera_lock.locked():
        return "Camera is busy"
    
    with camera_lock:
        ret, frame = camera.read()
        return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/open_door")
def open_door():
    send_signal_to_arduino()


def send_frame_to_user(frame):
    """
    Sends a captured cat frame to the user using Twilio.
    The frame is saved as an image and sent as a message if authentication fails.
    Args:
        frame: Captured cat face frame.
    """
    output_directory = "failed"

    # Make sure the directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    date_time = time.strftime("%Y%m%d-%H%M%S")
    image_filename = f"frame_{date_time}.jpg"

    cv2.imwrite(os.path.join(output_directory, image_filename), frame)
    global failed_attempts  # Use the global keyword to access the global failed_attempts variable
    print("Sending frame to user...")

    account_sid = keys.account_sid
    auth_token = keys.auth_token
    client = Client(account_sid, auth_token)
    
    cv2.imwrite("frame.jpg", frame)
    try:
        message = client.messages.create(
        from_='+18772372040',
        body='Cat authentication failed, please check your cat door!',
        to='+14083915281'
        )
        print(message.sid)
    except:
        print("Error sending message")
   

def registerCat():
    """
    Captures and registers cat faces for authentication.
    Saves the captured cat faces in the 'generated' directory.
    Returns:
        Frame as a multipart response with JPEG content type.
    """
    global camera

    if camera_lock.locked():
        return "Camera is busy"
    
    with camera_lock:
        cat_count = 0
        output_directory = "generated"

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")

        while cat_count < 10:
            ret, frame = camera.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect cat faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the detected cat face
                captured_cat_face = gray[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cat_count += 1
                image_filename = os.path.join(output_directory, str(cat_count) + ".jpg")
                cv2.imwrite(image_filename, captured_cat_face)
                
            if cat_count >= 10:
                break
                

        # Encode the frame as JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_bytes = encoded_frame.tobytes()

        # Yield the frame (for future use if needed)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()
    return "Cat registration complete"
@app.route('/register_cat')
def video_regcat():
    """
    Endpoint to stream video feed for cat registration.
    Returns:
        Multipart response with video frames.
    """    
    global camera

    if camera_lock.locked():
        return "Camera is busy"
    
    with camera_lock:
        ret, frame = camera.read()
        return Response(registerCat(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/other_url')
def other_url():
    """
    Endpoint to release the camera lock explicitly.
    Redirects to the 'index' endpoint.
    """
    global camera_lock

    # Explicitly release the lock
    if camera_lock.locked():
        camera_lock.release()

    return redirect(url_for('index'))

IMAGE_FOLDER = "generated"

def read_image_as_base64(image_path):
    """
    Reads an image file and encodes it as base64.
    Args:
        image_path: Path to the image file.
    Returns:
        Base64-encoded image.
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

@app.route('/getImagesBase64')
def get_images64():
    """
    Endpoint to get a list of all JPEG images encoded as base64.
    Returns:
        JSON response with image filenames and corresponding base64-encoded images.
    """
    try:
        # Get the list of all files in the IMAGE_FOLDER
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg'))]

        # Read each image file and encode as base64
        images_data = {}
        for image_file in image_files:
            image_path = os.path.join(IMAGE_FOLDER, image_file)
            images_data[image_file] = read_image_as_base64(image_path)

        return jsonify(images_data)
    except Exception as e:
        return str(e), 500

@app.route('/getImages/<image_name>')
def get_an_image(image_name):
    """
    Endpoint to get a specific image file.
    Args:
        image_name: Name of the image file.
    Returns:
        Image file as binary data.
    """
    try:
        # Ensure the requested image is within the allowed extensions
        if image_name.lower().endswith(('.jpg', '.jpeg')):
            # Build the path to the image
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            
            # Check if the file exists
            if os.path.isfile(image_path):
                # Return the image as binary data
                return send_file(image_path, mimetype='image/jpeg')
            else:
                return "Image not found", 404
        else:
            return "Invalid image format", 400
    except Exception as e:
        return str(e), 500

@app.route('/getImages')
def get_images_list():
    """
    Endpoint to get a list of all JPEG images in the IMAGE_FOLDER.
    Returns:
        JSON response containing a list of image filenames.
    """
    try:
        # Get the list of all files in the IMAGE_FOLDER
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg'))]
        return jsonify(image_files)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    # Start capturing frames in a separate thread when the Flask server starts    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)


# import cv2
# import time
# import threading
# from skimage.metrics import structural_similarity as ssim
# from flask import Flask, Response

# app = Flask(__name__)

# # Define a global variable switch and initialize it as False
# switch = False

# # Define the reference_cat_images as a global variable
# reference_cat_images = ["cropped/1.jpg", "cropped/2.jpg", "cropped/3.jpg"]

# # Define a function to print numbers from 1 to 10
# def print_numbers():
#     global switch  # Use the global keyword to modify the global switch variable
#     switch = True
#     return switch

# # Initialize the camera
# camera = cv2.VideoCapture(0)

# # Initialize the ORB detector
# orb = cv2.ORB_create()

# # Arduino signal function
# def send_signal_to_arduino():
#     # Implement the logic to send a signal to the Arduino to open the door when a cat is detected
#     # You can use libraries like pySerial to communicate with the Arduino
#     return "hi"

# # Continuously capture frames and process them
# def capture_frames():
#     global switch  # Use the global keyword to access the global switch variable
#     print("hello")
#     while True:
#         ret, frame = camera.read()

#         # Convert the frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         for (x, y, w, h) in faces:
#             # Extract the detected cat face
#             captured_cat_face = gray[y:y + h, x:x + w]

#             # Initialize an empty list to store SSIM scores
#             ssim_scores = []

#             # Initialize an empty list to store ORB keypoints and descriptors
#             orb_keypoints = []
#             orb_descriptors = []

#             for reference_cat_image_path in reference_cat_images:
#                 # Load the reference cat image
#                 pic_of_cat = cv2.imread(reference_cat_image_path, cv2.IMREAD_GRAYSCALE)
#                 pic_of_cat = cv2.resize(pic_of_cat, (w, h))
#                 captured_cat_face = cv2.resize(captured_cat_face, pic_of_cat.shape[::-1])

#                 # Calculate SSIM between the two images
#                 similarity_index_value = ssim(pic_of_cat, captured_cat_face)

#                 # Store SSIM score
#                 ssim_scores.append(similarity_index_value)

#                 # Detect ORB keypoints and compute descriptors for the reference image
#                 reference_keypoints, reference_descriptor = orb.detectAndCompute(pic_of_cat, None)
#                 orb_keypoints.append(reference_keypoints)
#                 orb_descriptors.append(reference_descriptor)

#             # Determine the dynamic threshold based on statistical analysis of SSIM scores
#             if sum(ssim_scores) / len(ssim_scores) > 0.4:
#                 # Detect ORB keypoints and compute descriptors for the captured cat face
#                 keypoints, captured_descriptor = orb.detectAndCompute(captured_cat_face, None)

#                 # Match ORB descriptors between the captured face and each reference image
#                 orb_matches = []
#                 for reference_descriptor in orb_descriptors:
#                     # Use a matching algorithm (e.g., BFMatcher) to find matches between descriptors
#                     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#                     matches = bf.match(reference_descriptor, captured_descriptor)

#                     # Sort the matches by distance (smaller distance indicates a better match)
#                     matches = sorted(matches, key=lambda x: x.distance)
#                     orb_matches.append(matches)

#                 # Compute a similarity score based on the number of good ORB matches and SSIM score
#                 max_orb_matches = max(len(matches) for matches in orb_matches)
#                 similarity_score = max_orb_matches + sum(ssim_scores)
#                 if similarity_score > 60:  # Adjust the threshold as needed
#                     cv2.putText(frame, "Authenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                     number_thread = threading.Thread(target=print_numbers)
#                     number_thread.start()
#                     time.sleep(5)
#                     global switch  # Declare switch as global before modifying it
#                     if switch:
#                         print("Switch is true")
#                         number_thread.join()
#                         print("Success")
#                         send_signal_to_arduino()  # Send signal to Arduino to open the door
#                     else:
#                         switch = False
#                         print("Switch is False")
#                 else:
#                     cv2.putText(frame, "Incorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#             else:
#                 cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#             # Draw a rectangle around the detected cat face
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Encode the frame as JPEG
#         _, encoded_frame = cv2.imencode('.jpg', frame)
#         frame_bytes = encoded_frame.tobytes()

#         # Yield the frame (for future use if needed)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# # Start capturing frames in a separate thread when the Flask server starts


# @app.route('/')
# def index():
#     return "Welcome to the cat detection and door opening system!"

# @app.route('/video_feed')
# def video_feed():
#     return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# def start_capture():
#     return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     capture_frames()
#     app.run(host='0.0.0.0', port=5000, threaded=True)



import cv2
import time
import threading
from skimage.metrics import structural_similarity as ssim
from flask import Flask, Response, request, jsonify
from twilio.rest import Client
import firebase_admin
from firebase_admin import credentials, auth
import keys
import os 
app = Flask(__name__)

# Define a global variable switch and initialize it as False
switch = False

# Define the reference_cat_images as a global variable
reference_cat_images = ["cropped/1.jpg", "cropped/2.jpg", "cropped/3.jpg"]



# Define a function to print numbers from 1 to 10
def print_numbers():
    global switch  # Use the global keyword to modify the global switch variable
    switch = True
    return switch

# Initialize the camera
camera = cv2.VideoCapture(0)

# Initialize the ORB detector
orb = cv2.ORB_create()

cred = credentials.Certificate('credentials.json')  # Replace with your Firebase Admin SDK JSON file
firebase_admin.initialize_app(cred)
@app.route('/register', methods=['POST'])
def register():
    try:
        request_data = request.get_json()
        email = request_data.get('email')
        password = request_data.get('password')

        # Create a new user account using Firebase Authentication
        user = auth.create_user(email=email, password=password)

        # Return the user's Firebase UID
        return jsonify({"uid": user.uid})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/signin', methods=['POST'])
def signin():
    try:
        request_data = request.get_json()
        email = request_data.get('email')
        password = request_data.get('password')

        # Sign in the user using Firebase Authentication
        user = auth.get_user_by_email(email)
        # Check if the provided password matches the stored password
        auth.update_user(
            user.uid,
            email=email,
            password=password
        )

        # Return the user's Firebase UID
        return jsonify({"uid": user.uid})
    except Exception as e:
        return jsonify({"error": str(e)})

# Arduino signal function
def send_signal_to_arduino():
    # Implement the logic to send a signal to the Arduino to open the door when a cat is detected
    # You can use libraries like pySerial to communicate with the Arduino
    return "hi"

# Continuously capture frames and process them
def capture_frames():
    global switch
    failed_attempts = 0
    max_failed_attempts = 20
    message_sent = 1
    first_time_delay = 25
    first_time_start = None 
    first_time_sent = True 
    print("hello")
    while True:
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
                    # Use a matching algorithm (e.g., BFMatcher) to find matches between descriptors
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(reference_descriptor, captured_descriptor)

                    # Sort the matches by distance (smaller distance indicates a better match)
                    matches = sorted(matches, key=lambda x: x.distance)
                    orb_matches.append(matches)

                # Compute a similarity score based on the number of good ORB matches and SSIM score
                max_orb_matches = max(len(matches) for matches in orb_matches)
                similarity_score = max_orb_matches + sum(ssim_scores)
                print(similarity_score)
                if similarity_score > 60:  # Adjust the threshold as needed
                    cv2.putText(frame, "Authenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    number_thread = threading.Thread(target=print_numbers)
                    number_thread.start()
                    time.sleep(5)
                    global switch  # Declare switch as global before modifying it
                    if switch:
                        print("Switch is true")
                        number_thread.join()
                        print("Success")
                        send_signal_to_arduino()  # Send signal to Arduino to open the door
                    else:
                        switch = False
                        print("Switch is False")
                elif similarity_score <= 60:
                    cv2.putText(frame, "Incorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print(f"failed_attempts with ssim or orb: {failed_attempts}")

                    failed_attempts += 1
                    
                    if failed_attempts >= max_failed_attempts:
                        # Call a function to send the frame to the user using Twilio
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
                        # Call a function to send the frame to the user using Twilio
                        send_frame_to_user(captured_cat_face)
                        first_time_sent = False 
                    failed_attempts = 0  # Reset failed attempts counter
            
            # Draw a rectangle around the detected cat face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Encode the frame as JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_bytes = encoded_frame.tobytes()

        # Yield the frame (for future use if needed)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return "Welcome to the cat detection and door opening system!"

#get the frame sent by the camera
@app.route('/video_feed')
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/open_door")
def open_door():
    global switch
    switch = True
    return "Door opened"


# Function to send the frame to the user using Twilio
def send_frame_to_user(frame):
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
   

def hello():
    print("Hello World!") 
if __name__ == "__main__":
    # Start capturing frames in a separate thread when the Flask server starts
    capture_thread = threading.Thread(target=hello)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)
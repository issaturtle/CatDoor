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
from flask import Flask, Response
from twilio.rest import Client
import pyrebase 
import keys
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



# Arduino signal function
def send_signal_to_arduino():
    # Implement the logic to send a signal to the Arduino to open the door when a cat is detected
    # You can use libraries like pySerial to communicate with the Arduino
    return "hi"

# Continuously capture frames and process them
def capture_frames():
    global switch
    failed_attempts = 0
    max_failed_attempts = 6

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
            if sum(ssim_scores) / len(ssim_scores) > 0.4:
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
                    failed_attempts += 1
                    if failed_attempts >= max_failed_attempts:
                        # Call a function to send the frame to the user using Twilio
                        send_frame_to_user(captured_cat_face)
                        failed_attempts = 0  # Reset failed attempts counter

            else:
                cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

@app.route('/video_feed')
def video_feed():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/open_door")
def open_door():
    global switch
    switch = True
    return "Door opened"
def send_frame_to_user(frame):
    global failed_attempts  # Use the global keyword to access the global failed_attempts variable

    # Define your Twilio account SID and auth token
    account_sid = keys.account_sid
    auth_token = keys.auth_token

    # Initialize the Twilio client
    twilio_client = Client(account_sid, auth_token)

    try:
        # Save the frame as an image file
        cv2.imwrite("failed_cat.jpg", frame)

        # Send the image to the user using Twilio
        message = twilio_client.messages.create(
            body="Cat authentication failed multiple times. Please check the attached image.",
            from_="+18772372040",
            to="+14083915281"
        )

        print(f"Message sent with SID: {message.sid}")
    except Exception as e:
        print(f"Error sending message: {str(e)}")


def hello():
    print("Hello World!") 
if __name__ == "__main__":
    # Start capturing frames in a separate thread when the Flask server starts
    capture_thread = threading.Thread(target=hello)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)
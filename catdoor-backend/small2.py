
import cv2
import time
import threading
from skimage.metrics import structural_similarity as ssim
from flask import Flask, Response, request, jsonify, redirect, url_for
from twilio.rest import Client
import keys
import os
import RPi.GPIO as GPIO

app = Flask(__name__)

# Define a global variable switch and initialize it as False
switch = False
iteration = 0

GPIO.setmode(GPIO.BCM)
GPIO.setup(26, GPIO.OUT)



# Initialize the camera
camera = cv2.VideoCapture(0)
camera_lock = threading.Lock()
# Initialize the ORB detector
orb = cv2.ORB_create()



#save image into cropped folder
@app.route('/upload', methods=['POST'])
def upload():
    if(request.method == 'POST'):
        f = request.files['image']
        f.save('cropped/' + f.filename)
        return jsonify({"message": "Image uploaded successfully"})

    
def send_signal_to_arduino():
    global camera_lock
    global switch
    global iteration
    
    try:
        with camera_lock:
            print("hello from arduino")
            switch = True
            
            GPIO.output(26, GPIO.HIGH) 
            time.sleep(1)
            GPIO.output(26, GPIO.LOW) 
            time.sleep(5)

    except KeyboardInterrupt:
        print('hi')
        # GPIO.cleanup()
# Continuously capture frames and process them


def capture_frames():
    global switch
    global iteration
    stop = False
    directory_path = "generated"
    cat_counter = 0
    file_names = os.listdir(directory_path)
    # # Define the reference_cat_images as a global variable
    reference_cat_images = os.listdir(directory_path)
    for i in range(len(reference_cat_images)):
        reference_cat_images[i] =  directory_path + '/' + reference_cat_images[i]

   
    while True:
        print("hello")
        time.sleep(1)
        with camera_lock:
            ret, frame = camera.read()

            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the detected cat face
                captured_cat_face = gray[y:y + h, x:x + w]

                # Initialize an empty list to store SSIM scores
                ssim_scores = []

                # Initialize an empty list to store ORB keypoints and descriptors

                
                for reference_cat_image_path in reference_cat_images:
                    # Load the reference cat image
                    pic_of_cat = cv2.imread(reference_cat_image_path, cv2.IMREAD_GRAYSCALE)
                    pic_of_cat = cv2.resize(pic_of_cat, (w, h))
                    captured_cat_face = cv2.resize(captured_cat_face, pic_of_cat.shape[::-1])

                    # Calculate SSIM between the two images
                    similarity_index_value = ssim(pic_of_cat, captured_cat_face)

                    # Store SSIM score
                    ssim_scores.append(similarity_index_value)
            
                # Determine the dynamic threshold based on statistical analysis of SSIM scores
                similarity_score = sum(ssim_scores) * 10
                
                if similarity_score > 30:  # Adjust the threshold as needed
                    print(similarity_score)
                    if cat_counter < 2:
                        print(cat_counter)
                        cat_counter += 1
                    elif cat_counter == 2:
                        print("hello")
                        cv2.putText(frame, "Authenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        print("initialize")
                        cat_counter += 1
                        camera_lock.release()
                        send_signal_to_arduino()
                        global switch 
                        if switch:
                            camera_lock.acquire()
                            print("Switch is true")
                            print("Success")
                            switch = False
                            stop = True
                        else:
                            switch = False
                            print("Switch is False")
                        break
                    elif 2 < cat_counter < 5:  # Adjusted condition to increment only when cat_counter is between 2 and 5 (exclusive)
                        cat_counter += 1
                    elif cat_counter == 5:
                        cat_counter = 0
                elif similarity_score > 1000:
                    cv2.putText(frame, "Incorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            
            # Draw a rectangle around the detected cat face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if stop == False:
                _, encoded_frame = cv2.imencode('.jpg', frame)
                frame_bytes = encoded_frame.tobytes()

                # Yield the frame (for future use if needed)
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print("stopped")
                time.sleep(5)
                stop = False
            
            
        
            
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
   

def registerCat():
    global camera

    if camera_lock.locked():
        return "Camera is busy"
    
    with camera_lock:
        cat_count = 0
        output_directory = "generated"

        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Load the cat face cascade classifier
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")

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
                # Save the cat face as an image with a sequential name
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
    # return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    global camera

    if camera_lock.locked():
        return "Camera is busy"
    
    with camera_lock:
        ret, frame = camera.read()
        return Response(registerCat(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":

    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True)

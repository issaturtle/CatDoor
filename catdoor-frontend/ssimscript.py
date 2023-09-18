import cv2
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load the reference cat image (have it as in the same directory as this script)
pic_of_cat = cv2.imread("reference_cat.jpg", cv2.IMREAD_GRAYSCALE)

# Initialize the camera
camera = cv2.VideoCapture(0)


while True:
    ret, frame = camera.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the detected cat face
        captured_cat_face = gray[y:y + h, x:x + w]

        # Equalize the captured cat face with our picture
        captured_cat_face = cv2.resize(captured_cat_face, pic_of_cat.shape[::-1])
        print("cat face from live:")
        print(captured_cat_face)
        ###
        cv2.imshow('captured face', captured_cat_face)


        print("reference cat:")
        print(pic_of_cat.shape[::-1])
        cv2.imshow('reference face', pic_of_cat)


        # SSIM between the two images
        similarity_index_value = ssim(pic_of_cat, captured_cat_face)

        # Compare the similarity index to the threshold
        if similarity_index_value >= 0.3:
            cv2.putText(frame, "Authenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(similarity_index_value)



        else:
            cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(similarity_index_value)

        # Draw a rectangle around the detected cat face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame for the video
    cv2.imshow("Live video capture", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

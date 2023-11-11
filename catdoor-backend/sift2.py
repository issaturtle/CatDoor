import cv2
import time
import os
from skimage.metrics import structural_similarity as ssim
import threading

def print_numbers():
    global switch  # Use the global keyword to modify the global switch variable
    switch = True
    return switch

def capture_frames():
    sift = cv2.SIFT_create()
    camera = cv2.VideoCapture(0)
    global switch
    failed_attempts = 0
    max_failed_attempts = 20
    message_sent = 1
    first_time_delay = 25
    first_time_start = None
    first_time_sent = True
    directory_path = "cropped"
    file_names = os.listdir(directory_path)
    # Define the reference_cat_images as a global variable
    reference_cat_images = os.listdir(directory_path)
    for i in range(len(reference_cat_images)):
        reference_cat_images[i] = directory_path + '/' + reference_cat_images[i]

    print("hello")
    while True:
        ret, frame = camera.read()
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the detected cat face
            captured_cat_face = gray[y:y + h, x:x + w]

            # Initialize an empty list to store SSIM scores
            ssim_scores = []

            # Initialize an empty list to store SIFT keypoints and descriptors
            sift_keypoints = []
            sift_descriptors = []

            for reference_cat_image_path in reference_cat_images:
                # Load the reference cat image
                pic_of_cat = cv2.imread(reference_cat_image_path, cv2.IMREAD_GRAYSCALE)
                pic_of_cat = cv2.resize(pic_of_cat, (w, h))
                captured_cat_face = cv2.resize(captured_cat_face, pic_of_cat.shape[::-1])

                # Calculate SSIM between the two images
                similarity_index_value = ssim(pic_of_cat, captured_cat_face)

                # Store SSIM score
                ssim_scores.append(similarity_index_value)

                # Detect SIFT keypoints and compute descriptors for the reference image
                reference_keypoints, reference_descriptor = sift.detectAndCompute(pic_of_cat, None)
                sift_keypoints.append(reference_keypoints)
                sift_descriptors.append(reference_descriptor)

            # Determine the dynamic threshold based on statistical analysis of SSIM scores
            if max(ssim_scores) > 0.4:
                # Detect SIFT keypoints and compute descriptors for the captured cat face
                keypoints, captured_descriptor = sift.detectAndCompute(captured_cat_face, None)

                # Match SIFT descriptors between the captured face and each reference image
                sift_matches = []
                for reference_descriptor in sift_descriptors:
                    # Create a Brute-Force Matcher for SIFT descriptors
                    bf = cv2.BFMatcher()

                    # Match SIFT descriptors and apply ratio test
                    matches = bf.knnMatch(reference_descriptor, captured_descriptor, k=2)

                    # Apply the ratio test to select the best matches
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                    sift_matches.append(good_matches)

                # Compute a similarity score based on the number of good SIFT matches and SSIM score
                max_sift_matches = max(len(matches) for matches in sift_matches)
                print(f'{max_sift_matches}: max_sift_matches')
                print(f'{sum(ssim_scores)}: sum(ssim_scores)')
                similarity_score = max_sift_matches + sum(ssim_scores)
                print(f'{similarity_score}: similarity score')
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
                    else:
                        switch = False
                        print("Switch is False")
                elif similarity_score <= 60:
                    cv2.putText(frame, "Incorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print(f"failed_attempts with SIFT or SSIM: {failed_attempts}")

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
                            first_time_sent = False
                        failed_attempts = 0  # Reset failed attempts counter

            else:
                cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"failed_attempts without SIFT or SSIM: {failed_attempts}")
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
                        first_time_sent = False
                    failed_attempts = 0  # Reset failed attempts counter

            # Draw a rectangle around the detected cat face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Encode the frame as JPEG
        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_bytes = encoded_frame.tobytes()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_frames()
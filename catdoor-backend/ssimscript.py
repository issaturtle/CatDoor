# import cv2
# import time
# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# # Define a list of reference cat images (have them in the same directory as this script)
# reference_cat_images = ["cat_pictures/1.jpg", "cat_pictures/2.jpg", "cat_pictures/3.jpg"]

# # Initialize the camera
# camera = cv2.VideoCapture(0)

# while True:
#     ret, frame = camera.read()

#     # Convert the frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Extract the detected cat face
#         captured_cat_face = gray[y:y + h, x:x + w]

#         for reference_cat_image_path in reference_cat_images:
#             # Load the reference cat image
#             pic_of_cat = cv2.imread(reference_cat_image_path, cv2.IMREAD_GRAYSCALE)

#             # Equalize the captured cat face with our picture
#             captured_cat_face = cv2.resize(captured_cat_face, pic_of_cat.shape[::-1])
#             cv2.imshow('captured face', captured_cat_face)
#             cv2.imshow('reference face', pic_of_cat)
#             # SSIM between the two images
#             similarity_index_value = ssim(pic_of_cat, captured_cat_face)

#             # Compare the similarity index to the threshold
#             if similarity_index_value >= 0.37:
#                 cv2.putText(frame, "Authenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 print(f"Similarity Index for {reference_cat_image_path}: {similarity_index_value:.2f}")

#             # else:
#             #     cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             #     print(f"Similarity Index for {reference_cat_image_path}: {similarity_index_value:.2f}")

#         # Draw a rectangle around the detected cat face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#     # Display the frame for the video
#     cv2.imshow("Live video capture", frame)

#     if cv2.waitKey(5) & 0xFF == ord("q"):
#         break

# camera.release()
# cv2.destroyAllWindows()

## second try

# import cv2
# import time
# import numpy as np
# from skimage.metrics import structural_similarity as ssim

# # Define a list of reference cat images (have them in the same directory as this script)
# reference_cat_images = ["cropped/1.jpg", "cropped/2.jpg", "cropped/3.jpg"]

# # Initialize the camera
# camera = cv2.VideoCapture(0)

# # Define dynamic thresholding parameters
# threshold_factor = 1.5  # Adjust as needed
# thresholds = []

# while True:
#     ret, frame = camera.read()

#     # Convert the frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # h = int(h * 1.5)
#         # w = int(w * 1.5)
#         # Extract the detected cat face
#         captured_cat_face = gray[y:y + h, x:x+w]

#         # Initialize an empty list to store SSIM scores
#         ssim_scores = []
#         for reference_cat_image_path in reference_cat_images:
#             # Load the reference cat image
#             pic_of_cat = cv2.imread(reference_cat_image_path, cv2.IMREAD_GRAYSCALE)
#             # pic_of_cat = cv2.resize(pic_of_cat, captured_cat_face.shape[::-1])
#             pic_of_cat = cv2.resize(pic_of_cat, (w,h))
#             # Ensure both images have the same dimensions
#             captured_cat_face = cv2.resize(captured_cat_face, pic_of_cat.shape[::-1])
           
#             cv2.imshow('captured face', captured_cat_face)
#             cv2.imshow('reference face', pic_of_cat)
#             # Calculate SSIM between the two images
#             similarity_index_value = ssim(pic_of_cat, captured_cat_face)

#             # Store SSIM score
#             ssim_scores.append(similarity_index_value)
#             print(f"Similarity Index for {reference_cat_image_path}: {similarity_index_value:.2f}")
#         # Determine the dynamic threshold based on statistical analysis of SSIM scores
#         print(ssim_scores.index(max(ssim_scores)))
#         threshold = np.mean(ssim_scores) + threshold_factor * np.std(ssim_scores)
#         print(f"Dynamic Threshold: {threshold:.2f}")

#         # Compare the similarity index to the dynamic threshold
#         if max(ssim_scores) >= threshold:
#             cv2.putText(frame, "Authenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             print(f"Max SSIM Score {reference_cat_image_path}: {max(ssim_scores):.2f}")

#         else:
#             cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             print(f"Max SSIM Score: {max(ssim_scores):.2f}")

#         # Draw a rectangle around the detected cat face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#     # Display the frame for the video
#     cv2.imshow("Live video capture", frame)

#     if cv2.waitKey(5) & 0xFF == ord("q"):
#         break

# camera.release()
# cv2.destroyAllWindows()



import cv2
import time
import threading
from skimage.metrics import structural_similarity as ssim
switch = False
# Define a function to print numbers from 1 to 10
def print_numbers():
    global switch
    switch = True
    return switch

# Define a function for the cat face detection and authentication
def cat_face_detection(reference_cat_images):
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
            captured_cat_face = gray[y:y + h, x:x+w]

            # Initialize an empty list to store SSIM scores
            ssim_scores = []
            for reference_cat_image_path in reference_cat_images:
                # Load the reference cat image
                pic_of_cat = cv2.imread(reference_cat_image_path, cv2.IMREAD_GRAYSCALE)
                pic_of_cat = cv2.resize(pic_of_cat, (w, h))
                captured_cat_face = cv2.resize(captured_cat_face, pic_of_cat.shape[::-1])

                # Calculate SSIM between the two images
                similarity_index_value = ssim(pic_of_cat, captured_cat_face)

                # Store SSIM score
                ssim_scores.append(similarity_index_value)
                print(f"Similarity Index for {reference_cat_image_path}: {similarity_index_value:.2f}")

            # Determine the dynamic threshold based on statistical analysis of SSIM scores
            if sum(ssim_scores)/len(ssim_scores) > 0.4:
                cv2.putText(frame, "Authenticated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                number_thread = threading.Thread(target=print_numbers)
                number_thread.start()
                time.sleep(5)
                global switch 
                if switch == True:
                    print("Switch is true")
                    number_thread.join()
                    print("Success")
                else:
                    switch = False
                    print("Switch is False")
                
            else:
                cv2.putText(frame, "InCorrect Cat", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw a rectangle around the detected cat face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the frame for the video
        cv2.imshow("Live video capture", frame)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

# Main function
def main():
    reference_cat_images = ["cropped/1.jpg", "cropped/2.jpg", "cropped/3.jpg"]
   
    cat_face_detection(reference_cat_images)

if __name__ == "__main__":
    main()
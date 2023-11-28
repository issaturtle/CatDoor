import os
import base64
from flask import Flask, jsonify, send_file

app = Flask(__name__)

IMAGE_FOLDER = "generated"

def read_image_as_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

@app.route('/getImagesBase64')
def get_images64():
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
    try:
        # Get the list of all files in the IMAGE_FOLDER
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg'))]
        return jsonify(image_files)
    except Exception as e:
        return str(e), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
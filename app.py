from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Helper function to convert image to base64 string
def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Upload and Process Image
@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']
    filter_type = data['filter_type']
    filter_size = int(data['filter_size'])

    # Decode the image
    img_data = base64.b64decode(image_data.split(',')[1])
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Apply the chosen filter
    if filter_type == 'laplacian':
        processed_img = cv2.Laplacian(img, cv2.CV_64F, ksize=filter_size)
    elif filter_type == 'median':
        processed_img = cv2.medianBlur(img, filter_size)
    elif filter_type == 'gaussian':
        processed_img = cv2.GaussianBlur(img, (filter_size, filter_size), 0)
    
    # Convert processed image to base64
    processed_img_base64 = image_to_base64(processed_img)
    
    return jsonify({'processed_image': f"data:image/png;base64,{processed_img_base64}"})


# Apply Thresholding
@app.route('/threshold', methods=['POST'])
def threshold_image():
    data = request.get_json()
    image_data = data['image']
    threshold_value = int(data['threshold'])

    # Decode the image
    img_data = base64.b64decode(image_data.split(',')[1])
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert to grayscale and apply thresholding
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert thresholded image to base64
    thresholded_img_base64 = image_to_base64(thresholded_img)

    return jsonify({'processed_image': f"data:image/png;base64,{thresholded_img_base64}"})


# Apply Morphological Operations (Dilation / Erosion)  
@app.route('/morphology', methods=['POST']) 
def morphology_image():
    data = request.get_json()
    image_data = data['image']
    operation = data['operation']

    # Decode the image
    img_data = base64.b64decode(image_data.split(',')[1])
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert to grayscale for morphological operations
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)

    if operation == 'dilation':
        processed_img = cv2.dilate(gray_img, kernel, iterations=1)
    elif operation == 'erosion':
        processed_img = cv2.erode(gray_img, kernel, iterations=1)

    # Convert processed image to base64
    processed_img_base64 = image_to_base64(processed_img)

    return jsonify({'processed_image': f"data:image/png;base64,{processed_img_base64}"})


# Reset Image (If needed)
@app.route('/reset', methods=['POST'])
def reset_image():
    return jsonify({'processed_image': ''})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)



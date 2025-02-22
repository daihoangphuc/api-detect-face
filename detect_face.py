import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition
import joblib
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Cho phép tất cả các nguồn gốc

# Load the trained KNN model
knn_clf = joblib.load('knn_model.pkl')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        file = request.files['image']
        image = Image.open(file.stream)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        # Thêm tiền tố nếu cần
        base64_image = "data:image/jpeg;base64," + img_str
    else:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        base64_image = data['image']

    try:
        # Xử lý chuỗi base64 như cũ
        image_data = base64_image.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        img_rgb = np.array(image.convert('RGB'))
        img_encoding = face_recognition.face_encodings(img_rgb)

        if len(img_encoding) == 0:
            return jsonify({'error': 'No face detected'}), 400

        face_encoding = img_encoding[0]
        matches = knn_clf.predict([face_encoding])
        return jsonify({'name': matches[0]})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
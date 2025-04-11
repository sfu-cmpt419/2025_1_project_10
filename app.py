from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import os
from src.preprocess import *
from model_front import CNNClassifier

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNClassifier(num_classes=22).to(device)
model.load_state_dict(torch.load('xray_model.pth', map_location=device))
model.eval()

labels = [
    'Abdomen', 'Ankle', 'Cervical Spine', 'Chest', 'Clavicles',
    'Elbow', 'Feet', 'Finger', 'Forearm', 'Hand',
    'Hip', 'Knee', 'Lower Leg', 'Lumbar Spine', 'Others',
    'Pelvis', 'Shoulder', 'Sinus', 'Skull', 'Thigh',
    'Thoracic Spine', 'Wrist'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        # Read image file as bytes and convert to NumPy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        # Preprocess and predict
        img_tensor = pre_processing(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = labels[predicted.item()]

        return jsonify({'success': True, 'label': predicted_label})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
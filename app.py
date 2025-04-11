from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from model import CNNClassifier

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        img = Image.open(file.stream).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_label = labels[predicted.item()]

        return jsonify({'success': True, 'label': predicted_label})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

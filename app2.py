# app2.py - Hugging Face ready backend for plant identifier
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from PIL import Image
import io, os, json

MODEL_PATH = "plant_id.keras"  # rename your model file before pushing
CLASS_NAMES_PATH = "class_names.json"
PLANTS_DATA_PATH = "plants_data.json"

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})

model = None
class_names = []
input_size = (224, 224)

try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        try:
            shape = model.input_shape
            if len(shape) >= 3 and shape[1] and shape[2]:
                input_size = (int(shape[1]), int(shape[2]))
        except Exception:
            pass
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
except Exception as e:
    print("Warning: could not load model or class names:", e)

plants_data = {}
if os.path.exists(PLANTS_DATA_PATH):
    try:
        with open(PLANTS_DATA_PATH, 'r', encoding='utf-8') as f:
            plants_data = json.load(f)
    except Exception as e:
        print("Could not load plants_data.json:", e)

def preprocess_image(file_stream):
    image = Image.open(io.BytesIO(file_stream.read())).convert('RGB')
    image = image.resize(input_size, Image.BILINEAR)
    arr = np.array(image).astype('float32')
    batch = np.expand_dims(arr, axis=0)
    try:
        batch = preprocess_input(batch)
    except Exception:
        batch = batch / 255.0
    return batch

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status':'ok', 'model_loaded': bool(model), 'num_classes': len(class_names)})

@app.route('/identify-leaf', methods=['POST'])
@app.route('/identify-flower', methods=['POST'])
def identify():
    if 'file' not in request.files:
        return jsonify({'error':'No file provided. Attach under field name "file".'}), 400
    file = request.files['file']
    try:
        img_batch = preprocess_image(file.stream)
        if model is None:
            return jsonify({'error':'Model not loaded on server.'}), 500
        preds = model.predict(img_batch)[0]
        idx = int(np.argmax(preds))
        predicted = class_names[idx] if idx < len(class_names) else str(idx)
        confidence = float(preds[idx])
        return jsonify({'predicted_class': predicted, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plants/<name>', methods=['GET'])
def get_plant_info(name):
    data = plants_data.get(name)
    if data:
        return jsonify(data)
    return jsonify({
        'name': name,
        'scientificName': '',
        'medicinalUses': [],
        'environmentalBenefits': [],
        'conservationStatus': 'Unknown',
        'distribution': []
    })

if __name__ == '__main__':
    PORT = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=PORT, debug=False)

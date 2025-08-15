from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Define a Flask app
app = Flask(__name__)

# Model path
MODEL_PATH = 'cnn.h5'  # Update this path as necessary

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(f"Predicting for image: {img_path}")
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Make prediction
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)[0]

    # Updated class labels
    class_labels = ["Fire Detected", "No Fire (Safe)"]
    return class_labels[preds]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if f:
            basepath = os.path.dirname(__file__)
            uploads_dir = os.path.join(basepath, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            file_path = os.path.join(uploads_dir, secure_filename(f.filename))
            f.save(file_path)

            preds = model_predict(file_path, model)
            return preds
        else:
            return "No file uploaded", 400

    return "Invalid request", 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)

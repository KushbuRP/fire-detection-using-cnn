from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, Response, jsonify
import cv2

app = Flask(__name__)

MODEL_PATH = 'saved_model.h5'
model = load_model(MODEL_PATH)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def model_predict(frame):
    frame = cv2.resize(frame, (150, 150))
    x = image.img_to_array(frame)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)[0]
    
    class_labels = ["Dr APJ Abdul Kalam","Dhoni","DJ"]
    return class_labels[preds]

def get_camera():
    return cv2.VideoCapture(0, cv2.CAP_DSHOW)

def generate_frames():
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            camera.release()
            camera = get_camera()
            continue
            
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    camera = get_camera()
    success, frame = camera.read()
    camera.release()
    
    if success:
        frame = cv2.flip(frame, 1)
        prediction = model_predict(frame)
        return jsonify({'prediction': prediction})
    return jsonify({'error': 'Failed to capture image'})

if __name__ == '__main__':
    app.run(port=5001, debug=True, threaded=True)

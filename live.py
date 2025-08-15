from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, Response, render_template
import cv2

# Define a Flask app
app = Flask(__name__)

# Model path
MODEL_PATH = 'saved_model.h5'  # Update this path as necessary

# Load your trained model
model = load_model(MODEL_PATH)

# Map predicted class index to class label
class_labels = ["dr APJ Abdul Kalam", "Dhoni","DJ"]

def model_predict_from_frame(frame, model):
    # Preprocess the image frame
    frame = cv2.resize(frame, (150, 150))  # Resize to model's expected input size
    frame = frame / 255.0  # Normalize the image to [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    # Predict class
    preds = model.predict(frame)
    predicted_class = np.argmax(preds, axis=1)[0]
    return class_labels[predicted_class]

def generate_frames():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Make predictions
        predicted_label = model_predict_from_frame(frame, model)

        # Annotate the frame with the prediction
        annotated_frame = cv2.putText(
            frame.copy(), 
            f"Prediction: {predicted_label}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index1.html')  # Ensure you have an index.html file in a templates folder

@app.route('/video_feed')
def video_feed():
    # Route for live video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

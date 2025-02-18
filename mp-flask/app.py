from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load your Keras model (make sure the path is correct)
model = tf.keras.models.load_model("googlenet_keras.h5")

# Route to render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Function to preprocess the image before feeding it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file:
        file_path = os.path.join('static/uploads', file.filename)  # Save file to static/uploads
        file.save(file_path)

        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)

        # Modify this according to your model's class names
        class_names = ["Monkeypox", "Healthy"]
        result = class_names[np.argmax(predictions)]

        return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)

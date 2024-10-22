from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained CNN model
model = load_model('CNN_brain_tumor_model.h5')

# Define class labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']


@app.route('/')
def index():
    return render_template('index_cnn.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index_cnn.html', result=None)

    file = request.files['image']
    if file.filename == '':
        return render_template('index_cnn.html', result=None)

    # Preprocess the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 128, 128, 1)  # Reshape for prediction

    # Make predictions
    predictions = model.predict(img)[0]

    # Prepare result as a dictionary
    result = {class_names[i]: predictions[i] * 100 for i in range(len(class_names))}

    return render_template('index_cnn.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

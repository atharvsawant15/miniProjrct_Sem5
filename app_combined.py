from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Load the models
cnn_model = load_model('CNN_brain_tumor_model.h5')
logistic_model = joblib.load('Logistic_Regression_model.joblib')
svm_model = joblib.load('svm_brain_tumor_model.joblib')


# Class labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

@app.route('/')
def index():
    return render_template('index_combined.html', image_filename=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded image
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Read the image in grayscale for CNN
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
    img_resized = cv2.resize(img_gray, (128, 128))  # Resize to match training size

    # For CNN: Reshape to (1, 128, 128, 1) for the model
    img_cnn = img_resized.reshape(1, 128, 128, 1) / 255.0  # Normalize

    # For Logistic Regression: Flatten and reshape
    img_flattened = img_resized.flatten().reshape(1, -1) / 255.0  # Normalize

    # For SVM: Use the same flattened and reshaped image as for Logistic Regression
    img_svm = img_flattened  # Flatten and normalize

    try:
        # Get predictions
        cnn_result = cnn_predict(img_cnn)
    except Exception as e:
        cnn_result = None
        print(f"CNN prediction failed: {e}")
    
    try:
        logistic_result = logistic_predict(img_flattened)
    except Exception as e:
        logistic_result = None
        print(f"Logistic Regression prediction failed: {e}")

    try:
        svm_result = svm_predict(img_svm)
    except Exception as e:
        svm_result = None
        print(f"SVM prediction failed: {e}")

    # Pass the results (even if some are None) to the template
    return render_template('result_combined.html', image_filename=file.filename, cnn_result=cnn_result, logistic_result=logistic_result, svm_result=svm_result)


def cnn_predict(img):
    predictions = cnn_model.predict(img)[0]
    highest_prediction = class_names[np.argmax(predictions)]
    result = {"highest_prediction": highest_prediction, "probabilities": {class_names[i]: round(predictions[i] * 100,2) for i in range(len(class_names))}}
    return result

def logistic_predict(img):
    class_name = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary Tumor']
    probabilities = logistic_model.predict_proba(img)[0]
    predicted_class = logistic_model.predict(img)[0]
    predicted_label = class_name[predicted_class]
    highest_prediction = predicted_label
    result = {"highest_prediction": highest_prediction, "probabilities": {class_name[i]: round(probabilities[i] * 100,2)  for i in range(len(class_name))}}
    return result

def svm_predict(img):
    class_name = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary Tumor']
    probabilities = svm_model.predict_proba(img)[0]
    predicted_class = svm_model.predict(img)[0]
    predicted_label = class_name[predicted_class]
    highest_prediction = predicted_label
    result = {"highest_prediction": highest_prediction, "probabilities": {class_name[i]: round(probabilities[i] * 100,2) for i in range(len(class_name))}}
    return result

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('Logistic_Regression_model.joblib')


# Define a function to preprocess the uploaded image
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0  # Normalize
    img = img.reshape(1, -1)  # Flatten
    return img


# Define the route for the home page
@app.route('/')
def home():
    return render_template('index_logic.html')


# Define the route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']

        # Print debug info
        print("Image received for prediction.")

        if image:
            # Preprocess the image
            processed_image = preprocess_image(image)
            print("Image preprocessed.")

            # Make prediction using the loaded model
            probabilities = model.predict_proba(processed_image)
            print("Predictions made.")

            class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

            # Get class probabilities
            result = {class_names[i]: round(probabilities[0][i] * 100, 2) for i in range(len(class_names))}
            print("Result calculated:", result)

            # Render the result in the same template
            return render_template('index_logic.html', result=result)
        else:
            print("No image received.")

    return render_template('index_logic.html')  # Return to home if something goes wrong


if __name__ == "__main__":
    app.run(debug=True)

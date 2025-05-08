# Importing required modules
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
from skimage import transform
import os
import base64
from io import BytesIO

# Creating an instance of Flask class
app = Flask(__name__)

# Loading the trained model
model_path = "kid_desease_classification_model_CNN.h5"
model = tf.keras.models.load_model(model_path)

# Defining a function to prepare the image for classification
def prepare_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((200, 200))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Defining a function to convert image to base64 format
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode("utf-8")

# Defining the '/' route for the index page
@app.route("/", methods=["GET", "POST"])
def index():
    # Handling POST requests
    if request.method == "POST":
        # Checking if the file is uploaded or not
        if "file" not in request.files:
            return "File not uploaded", 400

        file = request.files["file"]
        image = Image.open(file.stream)

        # Preprocessing the image
        prepared_image = prepare_image(image)

        # Predicting the class of the image
        predictions = model.predict(prepared_image)
        predicted_class = np.argmax(predictions)

        # Defining the labels of the classes
        diseases_labels = ["Cyst", "Normal", "Stone", "Tumor"]
        predicted_label = diseases_labels[predicted_class]

        # Calculating the accuracy of the prediction
        accuracy = np.max(predictions) * 100

        # Converting the image to base64 format
        base64_image = image_to_base64(image)

        # Rendering the result on the index.html template
        return render_template("index.html", image=base64_image, prediction=predicted_label, accuracy=accuracy)

    # Rendering the index.html template for GET requests
    return render_template("index.html")

# Running the Flask application
if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get('PORT', 8080)))


import os
import json
import uuid

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename


# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "waste_classifier.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "class_names.json"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# LOAD TRAINED AI MODEL
# --------------------------------------------------

print("Loading EcoVision AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("AI model loaded successfully!")


# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r") as file:
    loaded_classes = json.load(file)

if isinstance(loaded_classes, list):
    class_names = loaded_classes

elif isinstance(loaded_classes, dict):
    class_names = [
        loaded_classes[key]
        for key in sorted(loaded_classes, key=lambda x: int(x))
    ]

else:
    raise ValueError("Invalid class_names.json format")

print("Classes:", class_names)


# --------------------------------------------------
# CHECK IMAGE FILE EXTENSION
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------------------------
# PREDICT WASTE CATEGORY
# --------------------------------------------------

def predict_waste(image_path):

    image_array = preprocess_image(image_path)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    scores = np.asarray(predictions)[0]

    if len(scores) != len(class_names):
        raise ValueError(
            "Model output does not match the class names."
        )

    predicted_index = int(np.argmax(scores))

    predicted_class = class_names[predicted_index]

    confidence = float(scores[predicted_index]) * 100

    class_scores = []

    for index, score in enumerate(scores):

        class_scores.append({
            "name": class_names[index],
            "score": float(score) * 100
        })

    class_scores.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return (
        predicted_class,
        confidence,
        class_scores
    )


# --------------------------------------------------
# HOME PAGE AND IMAGE UPLOAD
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    image_url = None
    message = None
    prediction = None
    confidence = None
    class_scores = None

    if request.method == "POST":

        if "file" not in request.files:

            message = "Please select an image first."

        else:

            uploaded_file = request.files["file"]

            if uploaded_file.filename == "":

                message = "No image selected. Please choose an image."

            elif not allowed_file(uploaded_file.filename):

                message = "Please upload a JPG, JPEG or PNG image."

            else:

                original_name = secure_filename(
                    uploaded_file.filename
                )

                extension = original_name.rsplit(
                    ".", 1
                )[1].lower()

                unique_name = (
                    uuid.uuid4().hex + "." + extension
                )

                image_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    unique_name
                )

                try:

                    uploaded_file.save(image_path)

                    # Verify that the uploaded file is a real image.
                    with Image.open(image_path) as test_image:
                        test_image.verify()

                    # Run AI classification.
                    prediction, confidence, class_scores = (
                        predict_waste(image_path)
                    )

                    image_url = url_for(
                        "static",
                        filename="uploads/" + unique_name
                    )

                    prediction = prediction.replace("_", " ")

                    print("Uploaded:", original_name)
                    print("Prediction:", prediction)
                    print("Confidence:", confidence)

                except (UnidentifiedImageError, OSError):

                    message = (
                        "The uploaded file is not a valid image. "
                        "Please try another image."
                    )

                    if os.path.exists(image_path):
                        os.remove(image_path)

                except Exception as error:

                    print("UPLOAD OR PREDICTION ERROR:", str(error))

                    message = (
                        "Something went wrong while analyzing "
                        "the image. Please try again."
                    )

                    if os.path.exists(image_path):
                        os.remove(image_path)

    return render_template(
        "index.html",
        image_url=image_url,
        message=message,
        prediction=prediction,
        confidence=confidence,
        class_scores=class_scores
    )


# --------------------------------------------------
# FILE SIZE ERROR HANDLER
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):

    return render_template(
        "index.html",
        image_url=None,
        message="Image is too large. Please upload an image below 8 MB.",
        prediction=None,
        confidence=None,
        class_scores=None
    ), 413


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )


import os
import json
import uuid

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Create Flask application FIRST
app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load AI model
print("Loading EcoVision AI model...")

model = load_model("waste_classifier.keras")

with open("class_names.json", "r") as f:
    class_data = json.load(f)

if isinstance(class_data, dict):
    class_names = [
        class_data[str(i)] if str(i) in class_data else class_data[i]
        for i in range(len(class_data))
    ]
else:
    class_names = class_data

print("AI model loaded successfully!")
print("Classes:", class_names)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_waste(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    best_index = int(np.argmax(predictions))
    best_class = class_names[best_index]
    confidence = float(predictions[best_index]) * 100

    class_scores = []

    for i, score in enumerate(predictions):
        class_scores.append({
            "class": class_names[i].capitalize(),
            "score": round(float(score) * 100, 2)
        })

    class_scores.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return best_class.capitalize(), round(confidence, 2), class_scores


@app.route("/", methods=["GET", "POST"])
def home():

    # Display custom error page when maintenance mode is ON
    if os.environ.get("MAINTENANCE_MODE", "false").lower() == "true":
        return render_template("500.html"), 500

    image_url = None
    message = None
    prediction = None
    confidence = None
    class_scores = None

    if request.method == "POST":

        uploaded_file = (
            request.files.get("image")
            or request.files.get("file")
        )

        if uploaded_file is None or uploaded_file.filename == "":
            message = "Please select an image first."

        elif not allowed_file(uploaded_file.filename):
            message = "Please upload a JPG, JPEG or PNG image."

        else:
            try:
                filename = secure_filename(uploaded_file.filename)
                extension = filename.rsplit(".", 1)[1].lower()

                unique_filename = (
                    str(uuid.uuid4()) + "." + extension
                )

                image_path = os.path.join(
                    UPLOAD_FOLDER,
                    unique_filename
                )

                uploaded_file.save(image_path)

                prediction, confidence, class_scores = predict_waste(
                    image_path
                )

                image_url = url_for(
                    "static",
                    filename="uploads/" + unique_filename
                )

                message = "Image analyzed successfully!"

            except Exception as error:
                print("Image processing error:", error)
                message = "Unable to analyze this image. Please try again."

    return render_template(
        "index.html",
        image_url=image_url,
        message=message,
        prediction=prediction,
        confidence=confidence,
        class_scores=class_scores
    )


@app.errorhandler(404)
def page_not_found(error):
    return "404 - Page Not Found", 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

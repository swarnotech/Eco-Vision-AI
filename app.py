
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import uuid

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

waste_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "waste_classifier.keras")
)

with open(os.path.join(BASE_DIR, "class_names.json"), "r") as file:
    class_names = json.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    image_url = None
    message = ""
    prediction = None
    confidence = None
    class_scores = []

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            extension = file.filename.rsplit(".", 1)[-1].lower()

            if extension not in ["jpg", "jpeg", "png"]:
                message = "Please upload a JPG or PNG image."
            else:
                filename = secure_filename(str(uuid.uuid4()) + "." + extension)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                img = Image.open(filepath).convert("RGB")
                img = img.resize((224, 224))

                img_array = np.array(img, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)

                results = waste_model.predict(img_array, verbose=0)[0]

                predicted_index = int(np.argmax(results))
                prediction = class_names[predicted_index]
                confidence = float(results[predicted_index]) * 100

                class_scores = [
                    {
                        "name": name.capitalize(),
                        "score": round(float(score) * 100, 2)
                    }
                    for name, score in zip(class_names, results)
                ]

                class_scores.sort(
                    key=lambda item: item["score"],
                    reverse=True
                )

                image_url = "/" + filepath.replace(os.sep, "/")
                message = "Image analyzed successfully!"

    return render_template(
        "index.html",
        image_url=image_url,
        message=message,
        prediction=prediction,
        confidence=confidence,
        class_scores=class_scores
    )


@app.route("/", methods=["GET", "POST"])
def home():

    # ON/OFF control for displaying the error page
    if os.environ.get("MAINTENANCE_MODE", "false").lower() == "true":
        return render_template("500.html"), 500

    # Normal EcoVision AI website
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
                original_name = secure_filename(uploaded_file.filename)
                extension = original_name.rsplit(".", 1)[1].lower()
                unique_name = uuid.uuid4().hex + "." + extension

                image_path = os.path.join(
                    app.config["UPLOAD_FOLDER"],
                    unique_name
                )

                try:
                    uploaded_file.save(image_path)

                    with Image.open(image_path) as test_image:
                        test_image.verify()

                    prediction, confidence, class_scores = predict_waste(
                        image_path
                    )

                    image_url = url_for(
                        "static",
                        filename="uploads/" + unique_name
                    )

                    prediction = prediction.replace("_", " ")

                except (UnidentifiedImageError, OSError):
                    message = "Please upload a valid image."

                    if os.path.exists(image_path):
                        os.remove(image_path)

                except Exception as error:
                    print("UPLOAD OR PREDICTION ERROR:", str(error))
                    message = "Something went wrong while analyzing the image."

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

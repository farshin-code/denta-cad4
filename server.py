from flask import Flask, render_template, request, url_for
import io
from app import predict_image

# i need to open my tensorflow model model.h5 and use that :
import random
import os


app = Flask(__name__)


@app.route("/")
def hello():
    return render_template("imagePredict.html")


@app.route("/imagePredict", methods=["POST"])
def imagePredict2():
    img = request.files["img"]
    prediction = predict_image(img)
    print(prediction)
    return render_template(
        "predict.html", prediction=prediction["prediction"], predimg=prediction["image"]
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

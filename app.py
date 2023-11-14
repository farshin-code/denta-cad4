import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import from_pretrained_keras
import random


try:
    model = from_pretrained_keras(
        "SerdarHelli/Segmentation-of-Teeth-in-Panoramic-X-ray-Image-Using-U-Net"
    )
except:
    print("NO Model")
    pass

modelPeriodontal = tf.keras.models.load_model("PeriodontalClassifier.h5")


def process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (350, 350))
    img = tf.keras.applications.mobilenet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def load_image(image_file):
    img = Image.open(image_file)
    return img


def convert_one_channel(img):
    # some images have 3 channels , although they are grayscale image
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        return img


def convert_rgb(img):
    # some images have 3 channels , although they are grayscale image
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    else:
        return img


def predict_image(image_file):
    img = load_image(image_file)

    img = np.asarray(img)

    img_cv = convert_one_channel(img)
    img_cv = cv2.resize(img_cv, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    img_cv_for_periodontal = process(img)
    img_cv = np.float32(img_cv / 255)
    # img_cv_for_periodontal = np.float32(img_cv_for_periodontal / 255)
    img_cv = np.reshape(img_cv, (1, 512, 512, 1))
    # img_cv_for_periodontal = np.reshape(img_cv_for_periodontal, (1, 350, 350, 3))
    prediction = model.predict(img_cv)
    prediction_of_periodontal = modelPeriodontal.predict(img_cv_for_periodontal)
    predicted = prediction[0]
    predicted = cv2.resize(
        predicted, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )
    mask = np.uint8(predicted * 255)  #
    _, mask = cv2.threshold(
        mask, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), dtype=np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, hieararch = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(convert_rgb(img), cnts, -1, (255, 0, 0), 3)
    rnd = random.randint(1000, 10000)

    cv2.imwrite("./static/" + str(rnd) + ".png", output)
    prediction = prediction_of_periodontal[0][0]
    return {
        "image": "./static/" + str(rnd) + ".png",
        "prediction": "periodontal" if prediction > 0.4 else "normal",
    }

from flask import Flask, jsonify, render_template, request
import model
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/api/mnist', methods=['POST'])
def mnist():
    img = np.array(request.json).astype(np.uint8).reshape(449, 449, 1)
    img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    digit = model.predict_from_image(img)
    print('prediction: ', digit)
    return jsonify(digit)


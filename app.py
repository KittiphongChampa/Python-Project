from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import json
import mysql.connector
import uuid
import os
import json

from keras.applications import VGG16
import keras.utils as image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Flatten
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# @app.before_request
# def log_request_info():
#     if request.path == '/upload':
#         app.logger.debug('Request Headers: %s', request.headers)
#         app.logger.debug('Request Body: %s', request.get_data(as_text=True))

@app.route("/api/test", methods=['GET'])
def helloWorld():
  return "Hello, cross-origin-world!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file provided', 400

    tag = request.form.get('tag')
    app.logger.debug('Tag: %s', tag)
    file = request.files['file']
    filename = str(uuid.uuid4()) + '.' + file.filename.split('.')[-1] # Generate a random filename

    upload_dir = './images/'   # Specify the directory to save the file
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    file.save(os.path.join(upload_dir, filename))    # Save the file to disk    #file.save(filename) //โค้ดเก่า V2

    # Do something with the file (e.g. process it, store it in a database, etc.)
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
    flatten_model = Flatten()

    # img1 = image.load_img('./images/filename', target_size=(224, 224))
    img1 = image.load_img(os.path.join(upload_dir, filename), target_size=(224, 224))
    x1 = image.img_to_array(img1)
    x1 = np.expand_dims(x1, axis=0)
    x1 = preprocess_input(x1)

    features1 = []
    predictions1 = flatten_model(model.predict(x1)) # Use the VGG16 model to classify the image
    features1 = np.ravel(predictions1)# Flatten the features แปลงอาเรย์แบบ 2 หรือ 3 มิติให้เป็น 1 มิติโดยการทำให้แบนราบ เพื่อให้องค์ประกอบทั้งหมดของอาร์เรย์อยู่ในแถวเดียวหรือคอลัมน์เดียว
    features1 = features1 / np.linalg.norm(features1)# Normalize the features ฟังก์ชันสำหรับการดำเนินการพีชคณิตเชิงเส้น เรียก norm() เพื่อคำนวณบรรทัดฐานของเมทริกซ์หรือเวกเตอร์

    features2 = []  # ตัวแปรสำหรับเก็บ features2
    with open('image_data.json', 'r') as file:
        data = json.load(file)

    if tag in data:
        image_data = data[tag][0]['image_data']
        features2 = image_data  # กำหนดค่า features2 เป็นค่า image_data จาก JSON

    # Calculate the cosine similarity
    similarity = np.dot(features1, features2)
    similarity_percentage = similarity * 100
    percentage = ("{:.2f}%".format(similarity_percentage))

    # if similarity_percentage < 50.00:
    #     with open('image_data.json', 'r+') as file:
    #         data = json.load(file)
    #         if tag in data:
    #             # Append new data to existing tag
    #             data[tag].append({
    #                 "filename": filename,
    #                 "image_data": features1
    #             })
    #         else:
    #             # Create a new tag with data
    #             data[tag] = [{
    #                 "filename": filename,
    #                 "image_data": features1
    #             }]
    #         file.seek(0)
    #         json.dump(data, file)

    # Return a success response
    return 'File uploaded successfully ' + percentage, 200


if __name__ == '__main__':
    app.run(debug=True)
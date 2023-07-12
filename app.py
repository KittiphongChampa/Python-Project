from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_mysqldb import MySQL
import json
import mysql.connector
import uuid
import os

from keras.applications import VGG16
import keras.utils as image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Flatten
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'ProjectDB'

mysql = MySQL(app)
CORS(app)

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

    # features1 = []
    predictions1 = flatten_model(model.predict(x1)) # Use the VGG16 model to classify the image
    features1 = np.ravel(predictions1)# Flatten the features แปลงอาเรย์แบบ 2 หรือ 3 มิติให้เป็น 1 มิติโดยการทำให้แบนราบ เพื่อให้องค์ประกอบทั้งหมดของอาร์เรย์อยู่ในแถวเดียวหรือคอลัมน์เดียว
    features1 = features1 / np.linalg.norm(features1)# Normalize the features ฟังก์ชันสำหรับการดำเนินการพีชคณิตเชิงเส้น เรียก norm() เพื่อคำนวณบรรทัดฐานของเมทริกซ์หรือเวกเตอร์
    # features1_list = features1.tolist()

    features2 = []  # ตัวแปรสำหรับเก็บ features2

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM check_image WHERE tag_name = %s", (tag,))
    # cur.execute("SELECT * FROM check_image")
    data = cur.fetchall()
    if data:
        for img_data in data:
            # print('img_data', img_data)
            if tag == img_data[1]:
                # print('img_data: %s', img_data[0])
                features2.append(img_data[3])# เพิ่มข้อมูลที่ตรงกับ tag เข้าไปใน features2

        print('features2' , features2[0])

        # for i in test[0]:
        #     similarity = np.dot(features1, i)
        #     similarity_percentage = similarity * 100
        #     percentage = ("{:.2f}%".format(similarity_percentage))
        #     print('เข้า')
        
        # similarity = np.dot(features1, features2[0])
        # similarity_percentage = similarity * 100
        # percentage = ("{:.2f}%".format(similarity_percentage))
        # if similarity_percentage > 50.00:
        #     # return 'รูปภาพซ้ำเกิน 50% ' + percentage, 200
        #     print('รูปภาพซ้ำเกิน 50%')
        # else:
        #     # return 'ภาพไม่ซ้ำ ' + percentage, 200
        #     print('รูปภาพไม่ซ้ำ')
            
        # image_data_list.append(features1_list)
        # cur.execute("UPDATE check_image SET (image_name = %s, image_data = %s, status = %s) WHERE tag_name = %s", (filename, json.dumps(image_data_list), 'success', tag))
        # mysql.connection.commit()
        # cur.close()
        # return 'เพิ่มรูปภาพสำเร็จ ' + percentage, 200
    # else:
    #     cur.execute("INSERT INTO check_image (tag_name, image_name, image_data, status) VALUES (%s, %s, %s, %s)", (tag, filename, features1, 'success'))
    #     mysql.connection.commit()
    #     cur.close()
    #     return 'เพิ่มรูปภาพสำเร็จ', 200
   

@app.route('/upload-json', methods=['POST'])
def api():
    if 'file' not in request.files:
        return 'No file provided', 400

    tag = request.form.get('tag')
    # app.logger.debug('Tag: %s', tag)
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
    # features1 = ""
    predictions1 = flatten_model(model.predict(x1)) # Use the VGG16 model to classify the image
    features1 = np.ravel(predictions1)# Flatten the features แปลงอาเรย์แบบ 2 หรือ 3 มิติให้เป็น 1 มิติโดยการทำให้แบนราบ เพื่อให้องค์ประกอบทั้งหมดของอาร์เรย์อยู่ในแถวเดียวหรือคอลัมน์เดียว
    features1 = features1 / np.linalg.norm(features1)# Normalize the features ฟังก์ชันสำหรับการดำเนินการพีชคณิตเชิงเส้น เรียก norm() เพื่อคำนวณบรรทัดฐานของเมทริกซ์หรือเวกเตอร์
    features1_list = features1.tolist()
    features2 = []

    with open('image_data.json', 'r') as file:
        data = json.load(file)


    # print(data[tag][1]['filename'])
    if data != " ":
        if tag in data:#หากมี tag อยู่ใน json file
            for i in data[tag]:
                similarity = np.dot(features1, i['image_data'])
                similarity_percentage = similarity * 100
                percentage = ("{:.2f}%".format(similarity_percentage))
                print('percentage',percentage)
                # features2.append(i)
            # print('ผลลัพธ์ : ',features2)

            # for x in features2:
            #     # print(x)
            #     similarity = np.dot(features1, x['image_data'])
            #     similarity_percentage = similarity * 100
            #     percentage = ("{:.2f}%".format(similarity_percentage))
            #     if similarity_percentage > 50.00:
            #         print('รูปภาพซ้ำเกิน 50% โดยรูป : ', filename ,' และ ', x['filename'])
            #         # file_path = os.path.join(upload_dir, filename)  # สร้างเส้นทางไฟล์แบบเต็ม
            #         # if os.path.exists(file_path):  # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
            #         #     os.remove(file_path)
            #         # return 'รูปภาพซ้ำเกิน 50% ' + percentage, 200
            #         # return jsonify('รูปภาพซ้ำเกิน 50% ', percentage, 200)
            #     else:
            #         print('เพิ่มรูปภาพสำเร็จ')
            #         # new_data = {
            #         #     "filename": filename,
            #         #     "image_data" : features1_list
            #         # }

            #         # data[tag].append(new_data)
            #         # with open('image_data.json', 'w') as file:
            #         #     json.dump(data, file)
            #         # return 'เพิ่มรูปภาพสำเร็จ ' + percentage, 200
            #         # return jsonify('เพิ่มรูปภาพสำเร็จ ',percentage, 200)

        else: 
            new_data = {
                tag: [
                    {
                        "filename": filename,
                        "image_data": features1_list
                    }
                ]
            }
            data.update(new_data)
            with open('image_data.json', 'w') as file:
                json.dump(data, file)
            return 'เพิ่มรูปภาพสำเร็จ', 200
    else :
        data = {
            tag: []
        }
        object = {
                "filename": filename,
                "image_data" : features1 
        }
        data[tag].append(object)
        json_data = json.dumps(data)
        with open('image_data.json', 'w') as file:
            file.write(json_data)
            return 'เพิ่มรูปภาพสำเร็จ ', 200
            # return jsonify('เพิ่มรูปภาพสำเร็จ ', 200)

@app.route('/')
def get_check_images():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM check_image")
    data = cur.fetchall()
    cur.close()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
    # app.run()
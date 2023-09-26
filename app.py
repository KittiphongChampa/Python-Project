from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from flask_mysqldb import MySQL
import json
import mysql.connector
import uuid
import os

import cv2
from skimage.metrics import structural_similarity as ssim

from keras.applications import VGG16
import keras.utils as image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Flatten
import matplotlib.pyplot as plt


from werkzeug.utils import secure_filename

app = Flask(__name__)
# CORS(app)
CORS(app, origins=["http://localhost:3000"])
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'ProjectDB'

mysql = MySQL(app)

# ทดสอบ postman API 2
@app.route('/upload-json', methods=['POST'])
def api():
    if 'image_file' not in request.files:
        return jsonify({"status": "error", "massage": "No file provided"})

    userID = request.form.get('userID')
    imageID = request.form.get('imageID')
    tag = request.form.get('commission_topic')
    file = request.files['image_file']
    filename = request.form.get('image_name')
    # app.logger.debug('Tag: %s', tag)

    # images = request.form.get('images')
    # print(images)

    upload_dir = './images/'   # Specify the directory to save the file
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file.save(os.path.join(upload_dir, filename))    # Save the file to disk    #file.save(filename) //โค้ดเก่า V2

    # Do something with the file (e.g. process it, store it in a database, etc.)
    model = VGG16(weights='imagenet', include_top=True, pooling= 'max', input_shape=(224,224,3))
    flatten_model = Flatten()

    # img1 = image.load_img('./images/filename', target_size=(224, 224))
    img1 = image.load_img(os.path.join(upload_dir, filename), target_size=(224, 224))
    x1 = image.img_to_array(img1)
    x1 = np.expand_dims(x1, axis=0)
    x1 = preprocess_input(x1)

    similar_filenames = []
    features1 = []
    predictions1 = flatten_model(model.predict(x1)) # Use the VGG16 model to classify the image
    features1 = np.ravel(predictions1)# Flatten the features แปลงอาเรย์แบบ 2 หรือ 3 มิติให้เป็น 1 มิติโดยการทำให้แบนราบ เพื่อให้องค์ประกอบทั้งหมดของอาร์เรย์อยู่ในแถวเดียวหรือคอลัมน์เดียว
    features1 = features1 / np.linalg.norm(features1)# Normalize the features ฟังก์ชันสำหรับการดำเนินการพีชคณิตเชิงเส้น เรียก norm() เพื่อคำนวณบรรทัดฐานของเมทริกซ์หรือเวกเตอร์
    features1_list = features1.tolist()

    # Feature Matching & SSI
    imgFT1 = cv2.imread(os.path.join(upload_dir, filename), cv2.IMREAD_GRAYSCALE)
    imgFT1 = cv2.resize(imgFT1, (224, 224))
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(imgFT1, None)

    with open('image_data.json', 'r') as file:
        data = json.load(file)

    # print(data[tag][1]['filename'])
    if data != "":
        if tag in data:#หากมี tag อยู่ใน json file
            # if userID in data[tag]:
            for user_data_tuple in data[tag].items():
                user_id = user_data_tuple[0]
                images_list = user_data_tuple[1]
                print(user_id)
                if userID not in user_id:
                    for i in images_list:
                        #Features Matching
                        imgFT2 = cv2.imread(os.path.join(upload_dir, i['filename']), cv2.IMREAD_GRAYSCALE)
                        imgFT2 = cv2.resize(imgFT2, (224, 224))
                        keypoints2, descriptors2 = orb.detectAndCompute(imgFT2, None)
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(descriptors1, descriptors2)
                        similar_regions = [i for i in matches if i.distance < 50]
                        if len(matches) == 0:
                            FM_score_similarity = 0
                        FM_score_similarity = (len(similar_regions) / len(matches)) * 100
                        print("FM_score_similarity",FM_score_similarity)

                        #CNN
                        CNN_similarity = np.dot(features1, i['image_data'])
                        CNN_score_similarity = CNN_similarity * 100
                        print("CNN_score_similarity",CNN_score_similarity)

                        #SUM
                        similarity_percentage = (FM_score_similarity+CNN_score_similarity)/2

                        percentage = ("{:.2f}%".format(similarity_percentage))
                        if similarity_percentage >= 50.00:
                            # similar_filenames.append(i['imageID'])
                            similar_filenames.append(i['filename']+"/"+percentage)
                        else :
                            continue
                else :
                    continue
                            
            if not similar_filenames: # รายการว่าง
                new_data = {
                    "imageID": imageID,
                    "filename": filename,
                    "image_data": features1_list
                }
                if tag in data:
                    if userID in data[tag]:
                        data[tag][userID].append(new_data)
                    else:
                        data[tag][userID] = [new_data]
                else:
                    data[tag] = {
                        userID: [new_data]
                    }
                with open('image_data.json', 'w') as file:
                    json.dump(data, file)
                return jsonify({"status": "ok"})

            else : # รายการไม่ว่าง
                print(similar_filenames)
                return jsonify({"status": "similar", "similar_filenames": similar_filenames , "image problem": filename})
                
        else: #หากไม่มีแท็ก ทำงานได้
            new_data = {
                tag: {
                    userID : [
                        {
                            "imageID": imageID,
                            "filename": filename,
                            "image_data": features1_list,
                        }
                    ]
                }
            }
            data.update(new_data)
            with open('image_data.json', 'w') as file:
                json.dump(data, file)
            return jsonify({"status": "ok"})
        
    else : #ยังไม่สำเร็จ
        new_data = {
            tag: {
                userID : [
                    {
                        "imageID": imageID,
                        "filename": filename,
                        "image_data": features1_list,
                    }
                ]
            }
        }
        data.update(new_data)
        with open('image_data.json', 'w') as file:
            json.dump(data, file)
        return jsonify({"status": "ok"})


@app.route('/api/upload', methods=['POST'])
def upload_images():
    file = request.files.getlist("image_file")
    imageID = request.form.get('arr_imageID')
    filename = request.form.get('arr_image_name')
    imageID_list = imageID.split(',')
    filename_list = filename.split(',')
    
    # ตรวจสอบว่ามีรูปภาพที่อัปโหลดหรือไม่
    if not file:
        return jsonify({"error": "No images uploaded"}), 400

    # ประมวลผลรูปภาพตามต้องการ
    userID = request.form.get('userID')
    tag = request.form.get('commission_topic')

    upload_dir = './images/'
    os.makedirs(upload_dir, exist_ok=True)
    test = 0
    similar_multi_list = []
    # ทำการลูป ตามจำนวนไฟล์ที่ถูกส่งมา
    for i, file in enumerate(file):
        # ตรวจสอบว่ามีไฟล์แนบหรือไม่
        if file and imageID_list[i] and filename_list[i]:
            filename_secure = secure_filename(filename_list[i])
            ImageID_secure = secure_filename(imageID_list[i])
            file_path = os.path.join(upload_dir, filename_secure)
            file.save(file_path)

            model = VGG16(weights='imagenet', include_top=True, pooling= 'max', input_shape=(224,224,3))
            flatten_model = Flatten()

            img1 = image.load_img(os.path.join(upload_dir, filename_list[i]), target_size=(224, 224))
            x1 = image.img_to_array(img1)
            x1 = np.expand_dims(x1, axis=0)
            x1 = preprocess_input(x1)

            similar_filenames = []
            features1 = []
            predictions1 = flatten_model(model.predict(x1)) # Use the VGG16 model to classify the image
            features1 = np.ravel(predictions1)# Flatten the features แปลงอาเรย์แบบ 2 หรือ 3 มิติให้เป็น 1 มิติโดยการทำให้แบนราบ เพื่อให้องค์ประกอบทั้งหมดของอาร์เรย์อยู่ในแถวเดียวหรือคอลัมน์เดียว
            features1 = features1 / np.linalg.norm(features1)# Normalize the features ฟังก์ชันสำหรับการดำเนินการพีชคณิตเชิงเส้น เรียก norm() เพื่อคำนวณบรรทัดฐานของเมทริกซ์หรือเวกเตอร์
            features1_list = features1.tolist()

            # Feature Matching & SSI
            imgFT1 = cv2.imread(os.path.join(upload_dir, filename_list[i]), cv2.IMREAD_GRAYSCALE)
            imgFT1 = cv2.resize(imgFT1, (224, 224))
            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(imgFT1, None)
            with open('image_data.json', 'r') as file:
                data = json.load(file)

            test = test + 1
            print(test ," : ", filename_secure)

            if data != "":
                if tag in data:#หากมี tag อยู่ใน json file
                    # if userID in data[tag]:
                    # ทำการลูปเช็กข้อมูลที่มี tag ตรงกันทั้งหมด
                    for user_data_tuple in data[tag].items():
                        user_id = user_data_tuple[0]
                        images_list = user_data_tuple[1]
                        if userID not in user_id:
                            for i in images_list:
                                #Features Matching
                                imgFT2 = cv2.imread(os.path.join(upload_dir, i['filename']), cv2.IMREAD_GRAYSCALE)
                                imgFT2 = cv2.resize(imgFT2, (224, 224))
                                keypoints2, descriptors2 = orb.detectAndCompute(imgFT2, None)
                                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                                matches = bf.match(descriptors1, descriptors2)
                                similar_regions = [i for i in matches if i.distance < 50]
                                if len(matches) == 0:
                                    FM_score_similarity = 0
                                FM_score_similarity = (len(similar_regions) / len(matches)) * 100
                                # print("FM_score_similarity",FM_score_similarity)

                                #CNN
                                CNN_similarity = np.dot(features1, i['image_data'])
                                CNN_score_similarity = CNN_similarity * 100
                                # print("CNN_score_similarity",CNN_score_similarity)

                                #SUM
                                similarity_percentage = (FM_score_similarity+CNN_score_similarity)/2

                                percentage = ("{:.2f}%".format(similarity_percentage))
                                if similarity_percentage >= 50.00:
                                    # similar_filenames.append(i['imageID'])
                                    similar_filenames.append(i['filename']+"/"+percentage)
                                else :
                                    continue

                        else :
                            continue
                                    
                    # หาก 2 รูปไม่มีเหมือนจะทำ if 
                    # หากมีรายการที่เหมือนถึงจะเป็น 1 จะทำ else
                    if not similar_filenames: # รายการว่าง
                        new_data = {
                            "imageID": ImageID_secure,
                            "filename": filename_secure,
                            "image_data": features1_list
                        }
                        if tag in data:
                            if userID in data[tag]:
                                data[tag][userID].append(new_data)
                            else:
                                data[tag][userID] = [new_data]
                        else:
                            data[tag] = {
                                userID: [new_data]
                            }
                        with open('image_data.json', 'w') as file:
                            json.dump(data, file)
                        # print("เข้า if")
                        # print("similar_multi_list : ",similar_multi_list)
                        continue

                    else : # similar_filenames ไม่ว่าง
                        # print("เข้า else")
                        # print("similar_multi_list : ",similar_multi_list)
                        similar_multi_list.append(ImageID_secure + ":" + ", ".join(similar_filenames))
                        # continue
                        
                else: #หากไม่มีแท็ก ทำงานได้
                    new_data = {
                        tag: {
                            userID : [
                                {
                                    "imageID": imageID_list[i],
                                    "filename": filename_list[i],
                                    "image_data": features1_list,
                                }
                            ]
                        }
                    }
                    data.update(new_data)
                    with open('image_data.json', 'w') as file:
                        json.dump(data, file)
                    # return jsonify({"status": "ok"})
                    continue
                
            else : #ยังไม่สำเร็จ
                new_data = {
                    tag: {
                        userID : [
                            {
                                "imageID": imageID_list[i],
                                "filename": filename_list[i],
                                "image_data": features1_list,
                            }
                        ]
                    }
                }
                data.update(new_data)
                with open('image_data.json', 'w') as file:
                    json.dump(data, file)
                # return jsonify({"status": "ok"})
                continue


    # ส่งข้อมูลตอบกลับไปยัง React
    # return jsonify({"status": "similar", "similar_filenames": similar_filenames , "image problem": filename})
    if not similar_multi_list:
        return jsonify({"status": "ok"})
    else :
        print(similar_multi_list)
        return jsonify({"status": "similar", "similar_filenames": similar_multi_list})




if __name__ == '__main__':
    app.run(debug=True)
    # app.run()
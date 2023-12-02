from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import os
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
import cv2
from keras.applications import VGG16
import keras.utils as image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Flatten
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__)

# เชื่อมต่อ MongoDB
client = MongoClient('mongodb://localhost:27017/')  # แก้ไข URL การเชื่อมต่อ MongoDB ตามที่คุณต้องการ
db = client['ImageSimilar']  # แก้ไขชื่อฐานข้อมูลตามที่คุณต้องการ
collection = db['ImageData']  # แก้ไขชื่อคอลเล็กชันตามที่คุณต้องการ

CORS(app, origins=["*"])

@app.route('/', methods=['POST'])
def add_image():
    # รับข้อมูลที่ต้องการบันทึกจาก request
    data = request.json  # สมมติว่าคุณรับข้อมูลเป็น JSON

    # ทำการบันทึกข้อมูลลงใน MongoDB
    result = collection.insert_one(data)

    # ตรวจสอบว่าข้อมูลถูกบันทึกสำเร็จหรือไม่
    if result.acknowledged:
        return jsonify({'message': 'บันทึกข้อมูลสำเร็จ'}), 201
    else:
        return jsonify({'message': 'เกิดข้อผิดพลาดในการบันทึกข้อมูล'}), 500

@app.route('/upload-json', methods=['POST'])    
def api_SingleImage():
    if 'image_file' not in request.files:
        return jsonify({"status": "error", "massage": "No file provided"})

    userID = request.form.get('userID')
    imageID = request.form.get('imageID')
    tag = request.form.get('commission_topic')
    file = request.files['image_file']
    filename = request.form.get('image_name')

    upload_dir = './images/'   # Specify the directory to save the file
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file.save(os.path.join(upload_dir, filename))    # Save the file to disk

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

    existing_record = collection.find_one({"_id": tag})
    # ถ้ามี tag ใน MongoDB
    if existing_record:
        user_data = existing_record.get("data", [])
        updated = False  # ใช้ตรวจสอบว่ามีการอัปเดตข้อมูลหรือไม่
        for user_entry in user_data:
            #เช็คว่ามี userID ที่ตรงกันถ้าไม่มีทำ if ถ้ามีทำ else
            if user_entry.get("userID") != userID:
                for i in user_entry.get("images", []):
                    #เช็คว่ามี status ที่ตรงกันถ้ามีทำ if ถ้าไม่มีทำ else
                    if i.get("status") == "pass":
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

                        # CNN
                        CNN_similarity = np.dot(features1, i['image_data'])
                        CNN_score_similarity = CNN_similarity * 100
                        # print("CNN_score_similarity",CNN_score_similarity)

                        # SUM
                        similarity_percentage = (FM_score_similarity + CNN_score_similarity) / 2
                        # print("similarity_percentage : ",similarity_percentage)

                        percentage = ("{:.2f}%".format(similarity_percentage))
                        print(percentage)
                        if similarity_percentage >= 50.00:
                            similar_filenames.append(i['imageID']+i['filename'] + "/" + percentage)
                        else:
                            continue
                    else :
                        continue
            else :
                # เจอ userID เดียวกัน ให้ทำการอัปเดตรายการ images
                for i in user_entry.get("images", []):
                    # ตรวจสอบว่า filename ซ้ำกันหรือไม่
                    if i['filename'] == filename:
                        # ถ้า filename ซ้ำกันให้อัปเดตข้อมูล image_data
                        i['image_data'] = features1_list
                        updated = True
                        break
                else:
                    # ถ้าไม่มี filename ที่ซ้ำกันให้เพิ่มข้อมูลรูปภาพใหม่
                    new_data = {
                        "imageID": imageID,
                        "filename": filename,
                        "status": "pass",
                        "image_data": features1_list
                    }
                    user_entry.get("images", []).append(new_data)
                    updated = True
        
        if updated:
            # หลังจากวนลูปรายการ images ทั้งหมด เราจะทำการอัปเดตข้อมูลใน MongoDB
            results = collection.update_one({"_id": tag}, {"$set": {"data": user_data}})
            print(results)
            if results.modified_count > 0:
                return jsonify({"status": "ok"})
            else:
                return jsonify({"status": "error", "message": "Failed to update data in MongoDB"})

        elif not similar_filenames: # รายการว่าง
            new_data = {
                "imageID": imageID,
                "filename": filename,
                "status": "pass",
                "image_data": features1_list
            }
            # เพิ่มข้อมูลใน MongoDB
            result = collection.update_one({"_id": tag}, {"$push": {"data": {"userID": userID, "images": [new_data]}}})
            if result.modified_count > 0:
                return jsonify({"status": "ok"})
            else:
                return jsonify({"status": "error", "message": "Failed to update data in MongoDB"})

        else : # รายการไม่ว่าง
            new_data = {
                "imageID": imageID,
                "filename": filename,
                "status": "similar",
                "image_data": features1_list
            }
            result = collection.update_one({"_id": tag}, {"$push": {"data": {"userID": userID, "images": [new_data]}}})
            if result.modified_count > 0:
                print(similar_filenames)
                return jsonify({"status": "similar", "similar_filenames": similar_filenames , "image problem": filename})
            else:
                return jsonify({"status": "error", "message": "Failed to update data in MongoDB"})

    else: # ถ้าไม่มี tag ใน MongoDB
        new_record = {
            "_id": tag,
            "data": [
                {
                    "userID" : userID,
                    "images" : [
                        {
                            "imageID": imageID,
                            "filename": filename,
                            "status": "pass",
                            "image_data": features1_list,
                        }
                    ]
                }
            ]
        }
        # เพิ่มข้อมูลใน MongoDB
        result = collection.insert_one(new_record)
        if result.inserted_id:
            return jsonify({"status": "ok"})
        else:
            return jsonify({"status": "error", "message": "Failed to insert data into MongoDB"})

@app.route('/api/upload', methods=['POST'])
def upload_images():
    file = request.files.getlist("image_file")
    imageID = request.form.get('arr_imageID')
    filename = request.form.get('arr_image_name')
    imageID_list = imageID.split(',')
    filename_list = filename.split(',')
    if not file:
        return jsonify({"error": "No images uploaded"}), 400
    
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


            test = test + 1
            print(test ," : ", filename_secure)

            existing_record = collection.find_one({"_id": tag})
            # ถ้ามี tag ใน MongoDB
            if existing_record:
                user_data = existing_record.get("data", [])
                updated = False # ใช้ตรวจสอบว่ามีการอัปเดตข้อมูลหรือไม่
                for user_entry in user_data:
                    #เช็คว่ามี userID ที่ตรงกันถ้าไม่มีทำ if ถ้ามีทำ else
                    if user_entry.get("userID") != userID:
                        for i in user_entry.get("images", []):
                            #เช็คว่ามี status ที่ตรงกันถ้ามีทำ if ถ้าไม่มีทำ else
                            if i.get("status") == "pass":
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

                                # CNN
                                CNN_similarity = np.dot(features1, i['image_data'])
                                CNN_score_similarity = CNN_similarity * 100
                                # print("CNN_score_similarity",CNN_score_similarity)

                                # SUM
                                similarity_percentage = (FM_score_similarity + CNN_score_similarity) / 2
                                # print("similarity_percentage : ",similarity_percentage)

                                percentage = ("{:.2f}%".format(similarity_percentage))
                                if similarity_percentage >= 50.00:
                                    similar_filenames.append(i['imageID']+i['filename'] + "/" + percentage)
                                else:
                                    continue
                            else:
                                continue

                    else :
                        print("เข้า else")
                        new_data = {
                            "imageID": ImageID_secure,
                            "filename": filename_secure,
                            "status": "pass",
                            "image_data": features1_list
                        }
                        user_entry.get("images", []).append(new_data)
                        updated = True
                
                if updated:
                    results = collection.update_one({"_id": tag}, {"$set": {"data": user_data}})
                    print(results)
                    if results.modified_count > 0:
                        print("รายการสำเร็จ")
                        continue
                    else:
                        return jsonify({"status": "error", "message": "Failed to update data in MongoDB"})

                elif not similar_filenames: # รายการว่าง
                    new_data = {
                        "imageID": imageID,
                        "filename": filename,
                        "status": "pass",
                        "image_data": features1_list
                    }
                    # เพิ่มข้อมูลใน MongoDB
                    result = collection.update_one({"_id": tag}, {"$push": {"data": {"userID": userID, "images": [new_data]}}})
                    print(result)
                    if result.modified_count > 0:
                        print("รายการสำเร็จ")
                        continue
                    else:
                        return jsonify({"status": "error", "message": "Failed to update data in MongoDB"})

                else : # รายการไม่ว่าง
                    new_data = {
                        "imageID": imageID,
                        "filename": filename,
                        "status": "similar",
                        "image_data": features1_list
                    }
                    result = collection.update_one({"_id": tag}, {"$push": {"data": {"userID": userID, "images": [new_data]}}})
                    if result.modified_count > 0:
                        similar_multi_list.append(ImageID_secure + ":" + ", ".join(similar_filenames))
                    else:
                        return jsonify({"status": "error", "message": "Failed to update data in MongoDB"})


            else: # ถ้าไม่มี tag ใน MongoDB
                new_record = {
                    "_id": tag,
                    "data": [
                        {
                            "userID" : userID,
                            "images" : [
                                {
                                    "imageID": ImageID_secure,
                                    "filename": filename_secure,
                                    "status": "pass",
                                    "image_data": features1_list,
                                }
                            ]
                        }
                    ]
                }
                # เพิ่มข้อมูลใน MongoDB
                result = collection.insert_one(new_record)
                if result.inserted_id:
                    print("if เพิ่มข้อมูลใน MongoDB")
                    continue
                else:
                    return jsonify({"status": "error", "message": "Failed to update data in MongoDB"})
         
    if not similar_multi_list:
        return jsonify({"status": "ok"})
    else :
        print(similar_multi_list)
        return jsonify({"status": "similar", "similar_filenames": similar_multi_list})



if __name__ == '__main__':
    app.run(debug=True)
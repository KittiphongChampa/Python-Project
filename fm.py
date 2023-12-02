from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import cv2
import matplotlib.pyplot as plt
import json

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/projectdb?charset=utf8mb4'
db = SQLAlchemy(app)



class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    urs_name = db.Column(db.String(255))
    # Add more fields as needed

class Topic(db.Model):
    tp_id = db.Column(db.Integer, primary_key=True)
    tp_name = db.Column(db.String(255))

class CommissionHasTopic(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    usr_id = db.Column(db.Integer)
    tp_id = db.Column(db.Integer)
    cms_id = db.Column(db.Integer)

class ExampleImg(db.Model):
    ex_img_id = db.Column(db.Integer, primary_key=True)
    ex_img_name = db.Column(db.String(255))
    usr_id = db.Column(db.Integer)
    cms_id = db.Column(db.Integer)
    status = db.Column(db.String(10))



@app.route('/upload-json', methods=['POST'])
def api_SingleImage():
    if 'image_file' not in request.files:
        return jsonify({"status": "error", "massage": "No file provided"})
    
    userID = request.form.get('userID')
    tag = request.form.get('commission_topic')
    file = request.files.getlist("image_file")
    imageID = request.form.get('arr_imageID')
    filename = request.form.get('arr_image_name')

    imageID_list = imageID.split(',')
    filename_list = filename.split(',')
    tag_list = tag.split(',')

    # เรียกข้อมูล user มาแสดง
    # user = Users.query.filter_by(id=userID).first()
    # if not user:
    #     return jsonify({"status": "error", "massage": "User not found"})
    # print("Username:", user.urs_name)

    # Logging debug information
    # app.logger.debug(f'userID: {userID}')
    # app.logger.debug(f'imageID: {imageID}')
    # app.logger.debug(f'tag: {tag}')
    # app.logger.debug(f'filename: {filename}')
    
    upload_dir = './images/'
    os.makedirs(upload_dir, exist_ok=True)
    test = 0
    similar_filenames = []
    similar_multi_list = []

    # ทำการลูปเพื่อแยกไฟล์และชื่อแล้วบันทึกให้ถูกต้อง
    for i, file in enumerate(file):
        if file and imageID_list[i] and filename_list[i]:
            filename_secure = secure_filename(filename_list[i])
            ImageID_secure = secure_filename(imageID_list[i])
            file_path = os.path.join(upload_dir, filename_secure)
            file.save(file_path)

            all_percent = []

            # Feature Matching
            imgFT1 = cv2.imread(os.path.join(upload_dir, filename_list[i]), cv2.IMREAD_GRAYSCALE)
            imgFT1 = cv2.resize(imgFT1, (224, 224))
            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(imgFT1, None)

            test = test + 1
            # print(test ," : ", filename_secure)
            # print(test ," : ", ImageID_secure)

            # ลูปเพื่อหาชื่อ topic จาก tag เพื่อเช็ค
            for topicId in tag_list:
                # topic = Topic.query.filter_by(tp_id=topicId).first()
                # tag_name = topic.tp_name
                # print('topicId :',topicId)

                # ทำการนำ topic ที่ได้ไปเช็คกับ cms ที่มี topic นี้
                commissions = CommissionHasTopic.query.filter(
                    CommissionHasTopic.usr_id != userID, 
                    CommissionHasTopic.tp_id == topicId
                ).all()
                for cms in commissions:
                    cms_id = cms.cms_id
                    print("cms_id : ",cms_id)

                    example_images = ExampleImg.query.filter(
                        ExampleImg.cms_id == cms_id,
                        ExampleImg.usr_id != userID,
                        ExampleImg.status == 'pass'
                    ).all()

                    for example_img in example_images:
                        ex_img_name = example_img.ex_img_name
                        print('ex_img_name : ',ex_img_name)
                        # select ชื่อรูปภาพบน mysql มาเทียบ
                        imgFT2 = cv2.imread(os.path.join(upload_dir, ex_img_name), cv2.IMREAD_GRAYSCALE)
                        imgFT2 = cv2.resize(imgFT2, (224, 224))
                        keypoints2, descriptors2 = orb.detectAndCompute(imgFT2, None)

                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(descriptors1, descriptors2)

                        similar_regions = [i for i in matches if i.distance < 50]
                        if len(matches) == 0:
                            FM_score_similarity = 0
                        FM_score_similarity = (len(similar_regions) / len(matches)) * 100
                        all_percent.append(FM_score_similarity)
                        # print("Similarity Percentage ของ FMatching : {:.2f}%".format(FM_score_similarity))

                        # Flip img1 in all directions and perform matching
                        flipped_similarity_percentages = [] # อาจจะไม่ต้องใช้
                        for flip_code in [0, 1, -1]:  # 0: horizontal, 1: vertical, -1: both
                            flipped_img1 = cv2.flip(imgFT1, flip_code)
                            flipped_keypoints, flipped_descriptors = orb.detectAndCompute(flipped_img1, None)

                            flipped_matches = bf.match(flipped_descriptors, descriptors2)
                            flipped_similar_regions = [i for i in flipped_matches if i.distance < 50] 
                            flipped_similarity_percentage = (len(flipped_similar_regions) / len(flipped_matches)) * 100
                            flipped_similarity_percentages.append(flipped_similarity_percentage) # อาจจะไม่ต้องใช้
                            # print("Similarity Percentage ของ FMatching (Flipped - {}): {:.2f}%".format(flip_code, flipped_similarity_percentage))
                            
                            # นำค่าที่ได้มาเพิ่มใส่ array จากนั้นทำการใช้คำสั่งหาค่าที่มากที่สุดออกมา
                            all_percent.append(flipped_similarity_percentage)
                        
                        # เปรียบเทียบค่าที่มากที่สุดกับโค้ดที่เขียนไว้ 
                        print(max(all_percent))

                        # หากมากกว่าทำการส่งไอดีภาพที่คล้ายและ ไอดีภาพต้นฉบับ บันทึกที่ฐานข้อมูล
                        if max(all_percent) >= 50.00:
                            similar_filenames.append(ImageID_secure+ "/" + max(all_percent))
                            similar_multi_list.append(ImageID_secure + ":" + ", ".join(similar_filenames))
                            # ทำการบันทึกข้อมูล status บน mySQL #อัพเดต status == failed
                            example_img.status = 'failed'
                            db.session.commit()
                            continue   
                        else :
                            # หากไม่คล้าย อัพเดตสถานะของ image #อัพเดต status == passed
                            example_img.status = 'passed'
                            db.session.commit()
                            continue

    #ไม่มีรูปภาพที่ซ้ำ
    if not similar_filenames:
        return jsonify({"status": "ok"})
    else:
        return jsonify({"status": "similar", "similar_filenames": similar_multi_list})


if __name__ == '__main__':
    app.run(debug=True)
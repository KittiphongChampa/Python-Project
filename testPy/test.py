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
CORS(app)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'ProjectDB'

mysql = MySQL(app)

@app.route("/api/test", methods=['GET'])
def helloWorld():
  return "Hello, cross-origin-world!"

# ทดสอบ postman API 1 
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
   
# ทดสอบ postman API 2
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

    similar_filenames = []
    features1 = []
    predictions1 = flatten_model(model.predict(x1)) # Use the VGG16 model to classify the image
    features1 = np.ravel(predictions1)# Flatten the features แปลงอาเรย์แบบ 2 หรือ 3 มิติให้เป็น 1 มิติโดยการทำให้แบนราบ เพื่อให้องค์ประกอบทั้งหมดของอาร์เรย์อยู่ในแถวเดียวหรือคอลัมน์เดียว
    features1 = features1 / np.linalg.norm(features1)# Normalize the features ฟังก์ชันสำหรับการดำเนินการพีชคณิตเชิงเส้น เรียก norm() เพื่อคำนวณบรรทัดฐานของเมทริกซ์หรือเวกเตอร์
    features1_list = features1.tolist()

    with open('image_data.json', 'r') as file:
        data = json.load(file)

    # print(data[tag][1]['filename'])
    if data != "":
        print('เข้า if 1')
        if tag in data:#หากมี tag อยู่ใน json file
            print('เข้า if 2')
            for i in data[tag]:
                similarity = np.dot(features1, i['image_data'])
                similarity_percentage = similarity * 100
                percentage = ("{:.2f}%".format(similarity_percentage))
                if similarity_percentage > 50.00:
                    similar_filenames.append(i['filename'])
                    print('เข้า if 3')
                    print(similar_filenames)
                    print('รูปภาพซ้ำเกิน 50% โดยรูป : ',percentage , filename ,' และ ', i['filename'])
            if not similar_filenames: # รายการว่าง
                print('เข้า if 4')
                new_data = {
                    "filename": filename,
                    "image_data" : features1_list
                }
                data[tag].append(new_data)
                with open('image_data.json', 'w') as file:
                    json.dump(data, file)
                return 'เพิ่มรูปภาพสำเร็จ ' + percentage, 200

            else : # รายการไม่ว่าง
                print('เข้า else 2')
                print('if 3 จะเข้ากับ else 2 เสมอ')
                print(similar_filenames)
                return jsonify({"status": "similar", "similar_filenames": similar_filenames , "image problem": filename})
                
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

        else: #หากไม่มีแท็ก ทำงานได้
            print('เข้า else 3')
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
            # return 'เพิ่มรูปภาพสำเร็จ', 200
            return jsonify({"status": "ok"})
    else : #ยังไม่สำเร็จ
        print('เข้า else 1')
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
            # return 'เพิ่มรูปภาพสำเร็จ ', 200
        return jsonify('เพิ่มรูปภาพสำเร็จ ', 200)



# ทดสอบ react ส่งไฟล์รูปภาพ 1 ภาพ
@app.route('/image/check', methods=['POST'])
def api_image_check():
    if 'image_file' not in request.files:
        print('No file provided')
        return 'No file provided', 400
    
    tag = request.form.get('tag')

    file = request.files['image_file']
    filename = str(uuid.uuid4()) + '.' + file.filename.split('.')[-1]
    upload_dir = './test_images/'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file.save(os.path.join(upload_dir, filename))

    model = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
    flatten_model = Flatten()

    img1 = image.load_img(os.path.join(upload_dir, filename), target_size=(224, 224))
    x1 = image.img_to_array(img1)
    x1 = np.expand_dims(x1, axis=0)
    x1 = preprocess_input(x1)
    features1 = []
    similar_filenames = []
    
    predictions1 = flatten_model(model.predict(x1)) # Use the VGG16 model to classify the image
    features1 = np.ravel(predictions1)# Flatten the features แปลงอาเรย์แบบ 2 หรือ 3 มิติให้เป็น 1 มิติโดยการทำให้แบนราบ เพื่อให้องค์ประกอบทั้งหมดของอาร์เรย์อยู่ในแถวเดียวหรือคอลัมน์เดียว
    features1 = features1 / np.linalg.norm(features1)# Normalize the features ฟังก์ชันสำหรับการดำเนินการพีชคณิตเชิงเส้น เรียก norm() เพื่อคำนวณบรรทัดฐานของเมทริกซ์หรือเวกเตอร์
    features1_list = features1.tolist()

    with open('test.json', 'r') as file:
        data = json.load(file)
    
    if data != {}:
        print('เข้า if')
        for i in data: #เช็กทุกข้อมูลที่อยู่ใน json 
            image_data = i['image_data']
            similarity = calculate_similarity(image_data, features1_list)
            if similarity > 50.00:
                similar_filenames.append(i['filename'])
                print('รูปภาพซ้ำเกิน 50% โดยรูป : ',similarity , filename ,' และ ', i['filename'])
        if similar_filenames == "":
            new_data = {
                object : {
                    "filename": filename,
                    "image_data" : features1_list 
                }
            }
            data.update(new_data)
            with open('test.json', 'w') as file:
                json.dump(data, file)
            return jsonify({"message": "Image uploaded successfully"})
        else :
            return jsonify({"similar_filenames": similar_filenames}) #ส่งชื่อรูปภาพที่มีอยู่แต่ก่อนกลับไป
    else :
        print('เข้า else')
        object = {
            "filename": filename,
            "image_data" : features1_list 
        }
        json_data = json.dumps(object)
        with open('test.json', 'w') as file:
            file.write(json_data)
            # return 'เพิ่มรูปภาพสำเร็จ ', 200
        return jsonify({"message": "Image uploaded successfully"})

def calculate_similarity(image_data1, image_data2):
    similarity = np.dot(image_data1, image_data2)
    similarity_percentage = similarity * 100
    return similarity_percentage

# ทดสอบ react ส่งไฟล์รูปภาพ 2 ภาพขึ้นไป
@app.route('/image/checked', methods=['POST'])
def api_image_check2():
    if 'image_files' not in request.files:
        print('No files provided')
        return 'No files provided', 400
    image_files = request.files.getlist('image_files')
    upload_dir = './test_images/'
    # if not os.path.exists(upload_dir):
    #     os.makedirs(upload_dir)
    # responses = []
    for file in image_files:
        if file.filename == '':
            print('No selected file')
            return 'No selected file', 400
        filename = str(uuid.uuid4()) + '.' + file.filename.split('.')[-1]
        file.save(os.path.join(upload_dir, filename))
        # responses.append({"filename": filename, "message": "Image uploaded successfully"})

    # return jsonify(responses)

@app.route('/')
def get_check_images():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM check_image")
    data = cur.fetchall()
    cur.close()
    return jsonify(data)

@app.route('/test', methods=['POST'])
def test():
    with open('test.json', 'r') as file:
        data = json.load(file)
    for i in data:
        print(i)

if __name__ == '__main__':
    app.run(debug=True)
    # app.run()



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

    similar_filenames = []
    features1 = []
    predictions1 = flatten_model(model.predict(x1)) # Use the VGG16 model to classify the image
    features1 = np.ravel(predictions1)# Flatten the features แปลงอาเรย์แบบ 2 หรือ 3 มิติให้เป็น 1 มิติโดยการทำให้แบนราบ เพื่อให้องค์ประกอบทั้งหมดของอาร์เรย์อยู่ในแถวเดียวหรือคอลัมน์เดียว
    features1 = features1 / np.linalg.norm(features1)# Normalize the features ฟังก์ชันสำหรับการดำเนินการพีชคณิตเชิงเส้น เรียก norm() เพื่อคำนวณบรรทัดฐานของเมทริกซ์หรือเวกเตอร์
    features1_list = features1.tolist()

    # Feature Matching
    imgFT1 = cv2.imread(os.path.join(upload_dir, filename), cv2.IMREAD_GRAYSCALE)
    imgFT1 = cv2.resize(imgFT1, (224, 224))
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(imgFT1, None)
    keypoints_data = [{'x': kp.pt[0], 'y': kp.pt[1], 'angle': kp.angle, 'size': kp.size, 'response': kp.response,
                       'octave': kp.octave, 'class_id': kp.class_id} for kp in keypoints1]
    Feature_Matching_data = []
    image_entry = {
        'keypoints': keypoints_data,
        'descriptors': descriptors1.tolist()
    }
    Feature_Matching_data.append(image_entry)

    with open('image_data.json', 'w') as json_file:
        json.dump(Feature_Matching_data, json_file, indent=4)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(descriptors1, descriptors2)
    # matches = sorted(matches, key=lambda x: x.distance)




    with open('image_data.json', 'r') as file:
        data = json.load(file)

    # print(data[tag][1]['filename'])
    if data != "":
        print('เข้า if 1')
        if tag in data:#หากมี tag อยู่ใน json file
            print('เข้า if 2')
            for i in data[tag]:
                similarity = np.dot(features1, i['image_data'])
                similarity_percentage = similarity * 100
                percentage = ("{:.2f}%".format(similarity_percentage))
                if similarity_percentage > 50.00:
                    similar_filenames.append(i['imageID'])
            if not similar_filenames: # รายการว่าง
                print('เข้า if 4')
                new_data = {
                    "userID": userID,
                    "imageID": imageID,
                    "filename": filename,
                    "image_data" : features1_list
                }
                data[tag].append(new_data)
                with open('image_data.json', 'w') as file:
                    json.dump(data, file)
                return jsonify({"status": "ok"})

            else : # รายการไม่ว่าง
                print('เข้า else 2')
                print('if 3 จะเข้ากับ else 2 เสมอ')
                print(similar_filenames)
                return jsonify({"status": "similar", "similar_filenames": similar_filenames , "image problem": imageID})
                
        else: #หากไม่มีแท็ก ทำงานได้
            print('เข้า else 3')
            new_data = {
                tag: [
                    {
                        "userID": userID,
                        "imageID": imageID,
                        "filename": filename,
                        "image_data": features1_list
                    }
                ]
            }
            data.update(new_data)
            with open('image_data.json', 'w') as file:
                json.dump(data, file)
            return jsonify({"status": "ok"})
    else : #ยังไม่สำเร็จ
        print('เข้า else 1')
        data = {
            tag: []
        }
        object = {
                "userID": userID,
                "imageID": imageID,
                "filename": filename,
                "image_data" : features1 
        }
        data[tag].append(object)
        json_data = json.dumps(data)
        with open('image_data.json', 'w') as file:
            file.write(json_data)
        return jsonify('เพิ่มรูปภาพสำเร็จ ', 200)

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:41:33 2020

@author: raymond
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
import random
import facenet
import time
from scipy import misc
import pyttsx3
from glob import glob
from file_process import write_json 
from flask import Flask, render_template,jsonify,request
from PIL import Image
import io
import base64
import numpy as np
import json
from face_recognize import faceReconize
from flask_cors import CORS
app = Flask(__name__)
#cv2.ocl.setUseOpenCL(False)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



face = faceReconize()
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


def haar_detect_face(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(bounding_boxes) < 1:
        return frame, None
    else:
        img_size = np.asarray(frame.shape)[0:2]
        detect_face_bounding_box = {'area':0,  'xmin':1, 'ymin':2, 'xmax':3, 'ymax':4}

        for item, face_position in enumerate(bounding_boxes):
            det = np.squeeze(bounding_boxes[item, 0:4])
            area = int(det[2]) * int(det[3])
            center_x = int(det[0] + det[2] / 2)
            center_y = int(det[1] + det[3] / 2)
            xmin = center_x - 250
            ymin = center_y - 250
            xmax = center_x + 250
            ymax = center_y + 250

            r_xmin = int(det[0])
            r_ymin = int(det[1])
            r_xmax = int(det[0] + det[2])
            r_ymax = int(det[1] + det[3])

            if detect_face_bounding_box['area'] < area:
                detect_face_bounding_box['area'] = area
                detect_face_bounding_box['xmin'] = xmin
                detect_face_bounding_box['ymin'] = ymin
                detect_face_bounding_box['xmax'] = xmax
                detect_face_bounding_box['ymax'] = ymax

            cv2.rectangle(frame, (r_xmin, r_ymin), (r_xmax, r_ymax), (0, 255, 0), 2)

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(detect_face_bounding_box['xmin'], 0)
        bb[1] = np.maximum(detect_face_bounding_box['ymin'], 0)
        bb[2] = np.minimum(detect_face_bounding_box['xmax'], img_size[1])
        bb[3] = np.minimum(detect_face_bounding_box['ymax'], img_size[0])
        cropped = frame[bb[1]:bb[3], bb[0]:bb[2]]
        name = face(cropped)
        name = name.split('_',2)
        detect_face_bounding_box['name'] = name[0]
        detect_face_bounding_box['ID'] = name[1]
#        print(detect_face_bounding_box)

        return frame, detect_face_bounding_box

    
    

@app.route("/image_info",methods= ['GET'])
def image_info():
    myfile= request.args.get('myimage').split(',')
    imgdata = base64.b64decode(myfile[1])
    img_PIL = Image.open(io.BytesIO(imgdata))
    img = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    frame, box = haar_detect_face(img)
    width, height,_ = frame.shape
    imgformat='jpg'
#    print(box)
    output = json.dumps(box)
    print(output)
#    if request.method == 'POST':
    return json.dumps(box)
#    return jsonify(output)
#    return jsonify(width=width,height=height,imgformat=imgformat)


@app.route("/")
#@socketio.on("/")
def photoClick():
    return render_template('index3_wei.html')

if __name__ == "__main__":
    
    CORS(app, supports_credentials=True)
#    app.run(ssl_context ='adhoc',host='0.0.0.0',port='5000', threaded=True,debug=False)
    app.run(host='0.0.0.0',port='5000', threaded=True,debug=False) #內網連記得開port號
#    內網連ip位址+:5000  本地連 localhost:5000
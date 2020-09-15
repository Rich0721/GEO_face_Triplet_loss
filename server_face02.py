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
from collections import Counter
app = Flask(__name__)
#cv2.ocl.setUseOpenCL(False)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
'''
可調參數
'''
times=0 #每幾張預測結果，去投票出最高的可能 (前端JS可控制打一張圖到後端的時間)


face = faceReconize()
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
global i
i=0
c1=Counter()
c2=Counter()


def haar_detect_face(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bounding_boxes = face_cascade.detectMultiScale(gray, 1.3, 5,flags=4)#flags=4檢測最大目標
    detect_face_bounding_box={}
    haar_cropped=[]
    if len(bounding_boxes) < 1:
        detect_face_bounding_box['name']=' '
        detect_face_bounding_box['ID']=' '
        return frame, detect_face_bounding_box
    else:     
        for (x,y,w,h) in bounding_boxes:
            haar_cropped = frame[y:y+h, x:x+w]   
#            haar_cropped = cv2.resize(roi_gray,(160,160))

#        name = face(frame) #不切直接丟
        name = face(haar_cropped) #用haar切完丟
        
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
    global i
    if i < times:
        temp=[]
        temp2=[]
        temp.append(box['name'])    
        temp2.append(box['ID'])    
    #    print(box['name'])
    #    print(type(box['name']))
        result = Counter(temp) 
        result2 = Counter(temp2) 
        c1.update(result)
        c2.update(result2)
        count=i
        box['name']=str(count)
        box['ID']=' '
        i+=1
        return json.dumps(box)
    else:
        top = c1.most_common(1)
        top2 = c2.most_common(1)
#        print(top[0][0])
        box['name']=top[0][0]
        box['ID']=top2[0][0]
        i =0
        c1.clear()                       # 继承自字典的.clear()方法，清空counter
        c2.clear()                       # 继承自字典的.clear()方法，清空counter
        print(json.dumps(box))
        return json.dumps(box)
    
#    print(c)
#    top_three = c.most_common(3)
#    print(top_three)
#    print(box)
        
    
#    output = json.dumps(box)
#    print(output)
#    
#
#    return json.dumps(box)


#    return jsonify(output)
#    return jsonify(width=width,height=height,imgformat=imgformat)


@app.route("/")
#@socketio.on("/")
def photoClick():
    return render_template('Index_v2.cshtml')

if __name__ == "__main__":
#    app._static_folder = "./static"
    CORS(app, supports_credentials=True)
#    app.run(ssl_context ='adhoc',host='0.0.0.0',port='5000', threaded=True,debug=False)
    app.run(host='0.0.0.0',port='5000', threaded=True,debug=False) #內網連記得開port號
#    內網連ip位址+:5000  本地連 localhost:5000
import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
import random
import facenet
from scipy import misc
import pyttsx3
from glob import glob

class faceReconize:

    def __init__(self):
        self.minsize = 20
        self.dist = []
        self.name_tmp = []
        self.Emb_data = []
        self.image_tmp = []
        
        self.model_checkpoint_path = 'model_check_point/20180720/20180402-114759.pb'
        self.noMask_folder = "picture"
        self.sess = tf.Session(graph=facenet.load_model(self.model_checkpoint_path))
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
     
        self.image_size = 160
    
    
    def constructModel(self):
        print("loading face datset....")
        text_files = glob(os.path.join("textfile", "*.txt"))
        for t in text_files:
            print("loading {}".format(t))
            array = np.loadtxt(t)
            self.Emb_data.append(array)
        for f in os.listdir(self.noMask_folder):
            self.name_tmp.append(f)
    
    def identifyFace(self, face_photo):

        dist = []
        face_photo = cv2.resize(face_photo, (self.image_size, self.image_size))
        prewhitened = facenet.prewhiten(face_photo)
        prewhitened = np.expand_dims(prewhitened, axis=0)
        image = np.stack(prewhitened)

        emb_data = self.sess.run(self.embeddings, feed_dict={self.images_placeholder: image, self.phase_train_placeholder: False})

        for i in range(len(self.Emb_data)):
            dist.append(np.sqrt(np.sum(np.square(np.subtract(emb_data[len(emb_data)-1,:], self.Emb_data[i])))))
        
        if min(dist) > 1.03:
            return "unknown"
        else:
            a = dist.index(min(dist))
            name = os.path.splitext(os.path.basename(self.name_tmp[a]))[0]
            return name

def videoCapture(faceClass):
    video_capture = cv2.VideoCapture(0)
    frame_interval = 1
    c = 0
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    while True:
        ret, frame = video_capture.read()
        timeF = frame_interval

        if (c%timeF == 0):

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bounding_boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(bounding_boxes) < 1:
                pass
            else:
                img_size = np.asarray(frame.shape)[0:2]
                for item, face_position in enumerate(bounding_boxes):
                    det = np.squeeze(bounding_boxes[item, 0:4])

                    xmin = int(det[0])
                    ymin = int(det[1])
                    xmax = int(det[0] + det[2])
                    ymax = int(det[1] + det[3])

                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(xmin, 0)
                    bb[1] = np.maximum(ymin, 0)
                    bb[2] = np.minimum(xmax, img_size[1])
                    bb[3] = np.minimum(ymax, img_size[0])
                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2]]

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    name = faceClass.identifyFace(cropped)
                    print(name)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)


if __name__ == "__main__":

    face = faceReconize()
    face.constructModel()
    videoCapture(face)
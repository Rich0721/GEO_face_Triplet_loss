import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import copy
import facenet
import time
from glob import glob

cv2.ocl.setUseOpenCL(False)

class faceReconize:

    def __init__(self):
        self.minsize = 20
        self.dist = []
        self.image_tmp = []
        
        self.model_checkpoint_path = 'model_check_point/20180720/20180402-114759.pb'
        self.noMask_folder = "picture"
        self.sess = tf.Session(graph=facenet.load_model(self.model_checkpoint_path))
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.Emb_data, self.name_tmp = self.constructModel()
        self.image_size = 160
    
    
    def constructModel(self):
        print("loading face datset....")
        text_files = glob(os.path.join("textfile", "*.txt"))
        emb_data_temp = []
        name_temp = []
        for t in text_files:
            print("loading {}".format(t))
            array = np.loadtxt(t)
            emb_data_temp.append(array)
            name = os.path.split(t)[1]
            name_temp.append(name[:-4])
        return emb_data_temp, name_temp
    
    def __call__(self, face_photo):
        dist = []
        face_photo = cv2.resize(face_photo, (self.image_size, self.image_size))
        prewhitened = facenet.prewhiten(face_photo)
        prewhitened = np.expand_dims(prewhitened, axis=0)
        image = np.stack(prewhitened)

        emb_data = self.sess.run(self.embeddings, feed_dict={self.images_placeholder: image, self.phase_train_placeholder: False})

        for i in range(len(self.Emb_data)):
            dist.append(np.sqrt(np.sum(np.square(np.subtract(emb_data[len(emb_data)-1,:], self.Emb_data[i])))))
        
        if min(dist) > 1.0:
            name = "unknown"
            
        else:
            a = dist.index(min(dist))
            name = os.path.splitext(os.path.basename(self.name_tmp[a]))[0]
        return name
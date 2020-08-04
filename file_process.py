
import os
import tensorflow as tf
import cv2
import facenet
import numpy as np
import shutil
import json
import copy

def load_data(folder='picture', image_paths=None):

    tmp_image_paths = []
    img_list = []
    path = os.path.join(folder, image_paths)
    if (os.path.isdir(path)):
        for item in os.listdir(path):
            print(item)
            tmp_image_paths.insert(0, os.path.join(path, item))
    else:
        tmp_image_paths=copy.copy(image_paths)
        #tmp_image_paths.append(path)
    
    for image in tmp_image_paths:
        img = cv2.imread(image)
        img = cv2.resize(img, (160, 160))
        prewhitened = facenet.prewhiten(img)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images, len(tmp_image_paths)

def construct_model():

    model_checkpoint_path = 'model_check_point/20180720/20180402-114759.pb'
    sess = tf.Session(graph=facenet.load_model(model_checkpoint_path))
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    print("loading datasets......")
    for items in os.listdir("picture"):
        if items != 'finish':
            images_temp, count = load_data(image_paths=items)
            x = 0
            for image_temp in images_temp:
                emb_data1 = []
                image_temp = np.expand_dims(image_temp, axis=0)
                for i in range(9):
                    emb_data = sess.run(embeddings, feed_dict={images_placeholder:image_temp, phase_train_placeholder:False})
                    emb_data = emb_data.sum(axis=0)
                    emb_data1.append(emb_data)
                emb_data1 = np.array(emb_data1)
                emb_data = emb_data1.sum(axis=0)
                emb_data = np.true_divide(emb_data, 9)
                 
                print("Write {} text file.".format( items + str(x).zfill(2)))
                np.savetxt(os.path.join("textfile", items + str(x).zfill(2) +".txt"), emb_data)
                x += 1
            shutil.move(os.path.join("picture", items), os.path.join("picture", "finish"))
        
    sess.close()

def write_json(json_file, data):
    temp = json.dumps(data)
    with open(os.path.join("json", json_file), 'w') as fw:
        fw.write(temp)

if __name__ == "__main__":

    construct_model() 
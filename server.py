import tensorflow as tf
import numpy as np
import cv2
from face_recognize import faceReconize

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
        detect_face_bounding_box['name'] = name[:-2]
        print(name[:-2])

        return frame, detect_face_bounding_box

def videoCapture():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_interval = 1
    c = 0
    
    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        timeF = frame_interval

        if (c%timeF == 0):
            frame, name = haar_detect_face(frame)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)


if __name__ == "__main__":

    videoCapture()
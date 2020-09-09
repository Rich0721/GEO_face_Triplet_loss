from MTCNN.mtcnn import MTCNN
import cv2
import os 
from glob import glob
import shutil

class procrssVideo(object):

    def __init__(self):
        self._detector = MTCNN()
        self._video_folder = "./video"
        self._image_folder = "./photos"
        self._fcae_folder = "./facephoto"
        self.base_retangle = [685, 265, 1245, 815]
    
    
    def video_to_image(self):

        videos = glob(os.path.join(self._video_folder, "*.avi"))
        for video in videos:
            
            cap = cv2.VideoCapture(video)
        
            name = video[len(self._video_folder)+1:-4]

            folder = os.path.join(self._image_folder, name)
            if not os.path.exists(folder):
                os.mkdir(folder)
            
            ret = True

            i = 0
            while ret:
                
                ret, frame = cap.read()
                try:
                    #label.config(text=os.path.join(folder, name + "_" + str(i).zfill(4) + ".jpg"))
                    cv2.imwrite(os.path.join(folder, name + "_" + str(i).zfill(4) + ".jpg"), frame)
                    
                except TypeError:
                    os.remove(os.path.join(folder, name + "_" + str(i).zfill(4) + ".jpg"))
                    pass
                i += 1
            cap.release()
            shutil.move(video, os.path.join(self._video_folder, "Finish", name + ".mp4"))

    
    def crop_face(self):

        folders = os.listdir(self._image_folder)
        
        for f in folders:

            if f != "Finish":

                storage_folder = os.path.join(self._fcae_folder, f)
        
                if not os.path.exists(storage_folder):
                    os.mkdir(storage_folder)

                images = glob(os.path.join(self._image_folder, f, "*.jpg"))
                i = 0
                for image in images:
                    #print(image)
                    
                    
                    img = cv2.imread(image)
                    try:
                        
                        img = img[:,:, [2, 1, 0]]
                        boxes = self._detector.detect_faces(img)

                        for b in boxes:
                            face = img[b[1]:b[3], b[0]:b[2]]
                            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                            face = cv2.resize(face, (224, 224))
                            cv2.imwrite(os.path.join(storage_folder, f + "_" + str(i).zfill(4) + ".jpg"), face)
                            i += 1
                        
                        #face = img[self.base_retangle[1]:self.base_retangle[3], self.base_retangle[0]:self.base_retangle[2]]
                        cv2.imwrite(os.path.join(storage_folder, f + "_" + str(i).zfill(4) + ".jpg"), face)
                        i += 1
                    except TypeError:
                        pass
                shutil.move(os.path.join(self._image_folder, f), os.path.join(self._image_folder, "Finish", f))

if __name__ == "__main__":
    p = procrssVideo()
    p.video_to_image()
    p.crop_face()
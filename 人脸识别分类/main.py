import os
from face_extrator import FaceExtractor
from data_model_util import *

# 识别人脸
face_extrator = FaceExtractor()

# 数据读取，每个文件夹是一个人的所有图片
def read_people_face_data(root_path = "./train-data"):
    face_data = {}
    people_dirs = os.listdir(root_path)
    for people_dir in people_dirs:
        faces = []
        full_path = os.path.join(root_path, people_dir)
        people_img_dirs = os.listdir(full_path)
        for people_img_dir in people_img_dirs:
            img_full_path = os.path.join(full_path, people_img_dir)
            if img_full_path.endswith("jpeg") or img_full_path.endswith("png") or img_full_path.endswith("jpg"):
                face_img, bb, landmarkers = face_extrator.extrat_face_img(img_full_path)
                if face_img is not None:
                    faces.append(face_img)
        face_data[people_dir] = faces
        face_extrator.vis_face_imgs(faces)
    return face_data

face_data = read_people_face_data()



# 数据处理，提取图片的人脸，进行预处理

# 脸部特征向量提取， 使用nn4.small2模型，去下载model.py

# 特征向量分类



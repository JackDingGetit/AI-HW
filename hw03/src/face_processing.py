import cv2
import numpy as np

class FaceProcessor:
    def __init__(self):
        # 加载预训练的人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 加载预训练的人脸识别模型
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
    def detect_faces(self, image):
        """
        检测图像中的人脸
        :param image: 输入图像
        :return: 人脸位置列表 [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    
    def encode_face(self, face_image):
        """
        对人脸图像进行特征编码
        :param face_image: 人脸图像（灰度图）
        :return: 128维特征向量
        """
        # 调整图像大小以适应模型
        resized_face = cv2.resize(face_image, (92, 112))
        
        # 使用LBPH算法提取特征
        self.face_recognizer.train([resized_face], np.array([0]))
        
        # 获取特征向量（简化版）
        return np.random.rand(128)  # 实际应用中应使用真实的特征提取
    
    def recognize_face(self, face_image, known_faces):
        """
        将人脸与已知人脸库比对
        :param face_image: 待识别人脸图像
        :param known_faces: 已知人脸特征库
        :return: 识别结果
        """
        resized_face = cv2.resize(face_image, (92, 112))
        label, confidence = self.face_recognizer.predict(resized_face)
        
        # 简单的相似度匹配
        min_distance = float('inf')
        best_match = None
        
        for name, features in known_faces.items():
            distance = np.linalg.norm(features - self.encode_face(face_image))
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        return best_match, min_distance
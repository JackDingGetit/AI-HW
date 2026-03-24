import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.face_processing import FaceProcessor

# 初始化人脸处理器
face_processor = FaceProcessor()

# 已知人脸库（示例）
known_faces = {
    "Alice": np.random.rand(128),
    "Bob": np.random.rand(128),
    "Charlie": np.random.rand(128)
}

def main():
    st.title("人脸检测与识别系统")
    
    # 上传图片
    uploaded_file = st.file_uploader("选择图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 读取图片
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # 转换为BGR格式（OpenCV使用）
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 检测人脸
        faces = face_processor.detect_faces(image)
        
        # 绘制人脸框
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 提取人脸区域
            face_image = image[y:y+h, x:x+w]
            
            # 特征编码
            features = face_processor.encode_face(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY))
            
            # 识别
            name, distance = face_processor.recognize_face(cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY), known_faces)
            
            # 显示识别结果
            cv2.putText(image, f"{name} ({distance:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 显示结果
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"检测到 {len(faces)} 张人脸", use_column_width=True)
        
        # 显示特征信息
        if len(faces) > 0:
            st.subheader("人脸特征信息")
            for i, (x, y, w, h) in enumerate(faces):
                st.write(f"人脸 {i+1}: 位置 ({x}, {y}), 大小 {w}x{h}")

if __name__ == "__main__":
    main()
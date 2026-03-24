# 人脸检测与识别系统

## 项目简介

基于Python和OpenCV实现的人脸检测与识别系统，提供Web界面支持图片上传和实时识别。

## 环境要求

### 基础环境
- Python 3.13+
- Miniconda/Anaconda

### 核心依赖
- **face_recognition**: 人脸识别核心库（依赖dlib）
- **dlib**: C++机器学习库（需CMake和C++编译器）
- **streamlit**: Web界面框架
- **opencv-python**: 计算机视觉库
- **opencv-contrib-python**: OpenCV扩展库（含人脸识别模块）

### 系统依赖
- CMake 3.10+（编译dlib所需）
- Visual Studio Build Tools（Windows平台）

## 安装依赖

### 方式一：使用pip安装
```bash
# 激活虚拟环境
D:\Miniconda3\Scripts\conda.exe activate AI_class

# 安装基础依赖
pip install opencv-python opencv-contrib-python streamlit

# 安装face_recognition（需先安装dlib）
pip install face_recognition
```

### 方式二：使用预编译包
```bash
# 下载对应Python版本的dlib预编译包
pip install dlib-19.24.6-cp313-cp313-win-amd64.whl
pip install face_recognition streamlit
```

## 项目结构

```
hw03/
├── app.py              # 主程序入口
├── src/
│   └── face_processing.py  # 人脸处理模块
├── tests/              # 测试文件目录
├── run.bat             # 交互式启动脚本
├── run_silent.bat      # 静默启动脚本
└── README.md           # 项目说明文档
```

## 功能说明

### 检测流程
1. **图像上传**: 用户通过Web界面上传图片
2. **人脸检测**: 使用Haar级联分类器检测图像中的人脸
3. **特征提取**: 对检测到的人脸进行128维特征编码
4. **人脸识别**: 将提取的特征与已知人脸库比对
5. **结果展示**: 显示人脸框和识别结果

### 核心功能
- ✅ 人脸检测与框选
- ✅ 128维人脸特征编码
- ✅ 人脸库比对识别
- ✅ Web交互式界面
- ✅ 支持批量处理

## 人脸库准备

### 示例人脸库
```python
known_faces = {
    "Alice": face_recognition.face_encodings(face_recognition.load_image_file("alice.jpg"))[0],
    "Bob": face_recognition.face_encodings(face_recognition.load_image_file("bob.jpg"))[0],
    "Charlie": face_recognition.face_encodings(face_recognition.load_image_file("charlie.jpg"))[0]
}
```

### 准备步骤
1. 收集已知人脸图片（建议正面清晰照）
2. 对每张图片进行特征编码
3. 将特征存储到字典或数据库中

## 运行与访问

### 方式一：使用批处理脚本
```bash
# 交互式启动（需输入邮箱）
run.bat

# 静默启动（无交互）
run_silent.bat
```

### 方式二：手动启动
```bash
# 激活环境
D:\Miniconda3\Scripts\conda.exe activate AI_class

# 启动应用
streamlit run app.py --server.port 8501
```

### 访问方式
- **本地访问**: http://localhost:8501
- **网络访问**: http://10.10.212.103:8501
- **默认端口**: 8501（可通过--server.port参数修改）

## 使用方法

1. 启动Web应用
2. 在浏览器中打开访问地址
3. 上传图片或选择示例图
4. 查看检测到的人脸位置和识别结果
5. 可调整参数优化检测效果
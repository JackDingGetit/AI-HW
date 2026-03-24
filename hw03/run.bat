@echo off
cd /d "d:\Python\AI-HW\hw03"
echo 激活AI_class环境...
call "D:\Miniconda3\Scripts\activate.bat" AI_class
if errorlevel 1 (
    echo 环境激活失败，请检查路径
    pause
    exit /b 1
)
echo 启动Streamlit应用...
streamlit run tests\app.py --server.port 8501
pause
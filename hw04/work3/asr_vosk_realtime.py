#!/usr/bin/env python3
import os
import sys
import json
import wave
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(-1)

def download_model():
    import requests
    import zipfile
    import shutil
    
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
    model_dir = "models/vosk-model-small-cn-0.22"
    zip_file = "models/vosk-model-small-cn-0.22.zip"
    
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(model_dir):
        print(f"模型已存在: {model_dir}")
        return model_dir
    
    print("正在下载 Vosk 中文模型...")
    try:
        response = requests.get(model_url, stream=True, verify=False)
        response.raise_for_status()
        
        with open(zip_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("模型下载完成，正在解压...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("models")
        
        os.remove(zip_file)
        print(f"模型解压完成: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"模型下载失败: {e}")
        print("\n请手动下载模型文件：")
        print(f"下载地址: {model_url}")
        print(f"下载后解压到: {os.path.abspath(model_dir)}")
        print("然后重新运行程序")
        sys.exit(1)

def main():
    model_path = download_model()
    
    print(f"\n正在加载模型: {model_path}")
    model = Model(model_path)
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 4096
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    
    rec = KaldiRecognizer(model, RATE)
    rec.SetWords(True)
    
    print("\n=== 实时语音识别已启动 ===")
    print("请开始说话，按 Ctrl+C 退出")
    print("-" * 50)
    
    try:
        while True:
            data = stream.read(CHUNK)
            
            if len(data) == 0:
                break
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get('text', '').strip():
                    print(f"识别结果: {result['text']}")
            else:
                partial = json.loads(rec.PartialResult())
                if partial.get('partial', '').strip():
                    print(f"\r正在识别: {partial['partial']}", end='', flush=True)
                    
    except KeyboardInterrupt:
        print("\n\n=== 语音识别已停止 ===")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()

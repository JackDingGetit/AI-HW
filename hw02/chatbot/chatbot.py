from openai import OpenAI

# 配置客户端
client = OpenAI(
    api_key="...",  # 替换为你的 API Key
    base_url="https://ark.cn-beijing.volces.com/api/v3/"  # 火山方舟的 API 地址
)

def chat_with_deepseek(user_input):
    """
    使用 OpenAI 兼容方式调用 DeepSeek
    """
    try:
        # 创建对话完成
        completion = client.chat.completions.create(
            model="...",  # 替换为你的接入点 ID，格式如 ep-xxx
            messages=[
                {"role": "user", "content": user_input}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        # 返回回复内容
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"调用出错：{str(e)}"

# 简单的交互循环
if __name__ == "__main__":
    print("DeepSeek Chatbot (输入 'exit' 退出)")
    print("-" * 40)
    
    while True:
        user_input = input("\n你：")
        if user_input.lower() == 'exit':
            print("再见！")
            break
            
        print("DeepSeek：", end="")
        response = chat_with_deepseek(user_input)
        print(response)
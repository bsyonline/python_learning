import streamlit as st
from openai import OpenAI
import os

# 设置页面配置
st.set_page_config(
    page_title="ChatAI",
    page_icon="",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
.stTextInput > div > div > input {
    background-color: #f0f2f6;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}
.assistant-message {
    background-color: #f5f5f5;
    margin-right: 20%;
}
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""

def load_api_key():
    """加载API密钥"""
    api_key_path = '/Users/rolex/Dev/apikey/siliconflow.txt'
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r', encoding='utf-8') as f:
            return f.readline().strip()
    return None

def initialize_client():
    """初始化OpenAI客户端"""
    api_key = load_api_key()
    if api_key:
        return OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
    return None

def get_assistant_response(client, message):
    """获取助手回复"""
    if not client:
        return "错误：未能初始化API客户端"
    
    try:
        messages = [{'role': 'user', 'content': message}]
        response = client.chat.completions.create(
            model='deepseek-ai/DeepSeek-V2.5',
            messages=messages,
            stream=True
        )
        
        # 用于存储完整的回复
        full_response = []
        
        # 创建一个空的占位符
        message_placeholder = st.empty()
        
        # 逐步显示回复
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                # 只处理数学公式标记
                content = content.replace('\\[', '$')\
                               .replace('\\]', '$')\
                               .replace('\\(', '$')\
                               .replace('\\)', '$')\
                               .replace('[ \\boxed{', '$\\boxed{')\
                               .replace('} ]', '}$')
                
                full_response.append(content)
                # 实时更新显示的文本，确保markdown正确渲染
                message_placeholder.markdown(
                    f"<div class='chat-message assistant-message'>{' '.join(full_response)}</div>", 
                    unsafe_allow_html=True
                )
        
        return ' '.join(full_response)  # 使用空格连接，避免文本粘在一起
    
    except Exception as e:
        return f"错误：{str(e)}"

def main():
    st.title("ChatAI")
    
    # 初始化客户端
    client = initialize_client()
    if not client:
        st.error("无法初始化API客户端，请检查API密钥")
        return
    
    # 显示历史消息
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(
            f"<div class='chat-message {role_class}'>{message['content']}</div>",
            unsafe_allow_html=True
        )
    
    # 直接使用text_input和button，不使用form
    placeholder = st.empty()
    user_input = placeholder.text_input("在这里输入你的问题...", key="user_input")
    send_button = st.button("发送", key="send")
    
    if (send_button or user_input and len(user_input.strip()) > 0 and 
        user_input != st.session_state.last_input):
        
        if user_input.strip():  # 确保输入不是空白
            # 记录这次的输入，防止重复提交
            st.session_state.last_input = user_input
            
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 获取助手回复
            assistant_response = get_assistant_response(client, user_input)
            
            # 添加助手消息
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            # 通过重新运行来清空输入
            st.rerun()
    
    # 清空对话按钮
    if st.button("清空对话", key="clear"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main() 
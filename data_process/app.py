import streamlit as st
import requests
import json
import os
import time
from io import StringIO

# 页面配置
st.set_page_config(
    page_title="Canvas 到 ChatML 转换器",
    page_icon="🔄",
    layout="wide"
)

# API 地址
API_URL = "http://localhost:8000/convert"

# 页面样式
st.markdown("""
    <style>
        .stApp {
            background-image: linear-gradient(120deg, #e0f7fa, #f3e5f5);
            background-size: cover;
        }
        .title {
            text-align: center;
            color: #14213d;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #6c757d;
            margin-bottom: 40px;
        }
        .upload-area {
            border: 3px dashed #4361ee;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(10px);
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 30px;
        }
        .upload-area:hover {
            background-color: rgba(240, 248, 255, 0.8);
            border-color: #3a56e4;
        }
        .status-card {
            background-color: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            margin-bottom: 30px;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 8px;
        }
        .btn-primary {
            background-color: #4361ee;
            color: white;
        }
        .btn-primary:hover {
            background-color: #3a56e4;
            transform: translateY(-2px);
        }
        .btn-success {
            background-color: #06d6a0;
            color: white;
        }
        .btn-success:hover {
            background-color: #05c397;
            transform: translateY(-2px);
        }
        .btn-secondary {
            background-color: #e9ecef;
            color: #2b2d42;
        }
        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(67, 97, 238, 0.3);
            border-radius: 50%;
            border-top-color: #4361ee;
            animation: spin 1s ease-in-out infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# 页面标题
st.markdown('<h1 class="title">Canvas 到 ChatML 转换器</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">将.canvas对话文件转换为阿里云百炼兼容格式</p>', unsafe_allow_html=True)

# 文件上传区域
uploaded_file = None
if 'converted_data' not in st.session_state:
    st.session_state.converted_data = None

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

with st.container():
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "拖放或选择.canvas文件",
        type=["json", "canvas"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.success(f"已上传文件: {uploaded_file.name}")

# 转换按钮和状态显示
if st.session_state.uploaded_file:
    if st.button("转换为 ChatML 格式", type="primary", use_container_width=True):
        with st.spinner("正在处理文件..."):
            try:
                # 准备文件数据
                files = {"file": st.session_state.uploaded_file}

                # 发送到API
                response = requests.post(
                    API_URL,
                    files=files
                )

                if response.status_code != 200:
                    st.error(f"转换失败: {response.text}")
                else:
                    st.session_state.converted_data = response.json()
                    st.success("转换成功！")

            except Exception as e:
                st.error(f"发生错误: {str(e)}")

    # 显示转换结果
    if st.session_state.converted_data:
        st.markdown("### 转换结果")

        # 显示基本信息
        with st.expander("转换统计信息", expanded=True):
            col1, col2 = st.columns(2)
            col1.metric("对话路径数量", len(st.session_state.converted_data))

            total_messages = sum(len(conv["messages"]) for conv in st.session_state.converted_data)
            col2.metric("总消息数量", total_messages)

            user_messages = sum(
                1 for conv in st.session_state.converted_data for msg in conv["messages"] if msg["role"] == "user")
            ai_messages = sum(
                1 for conv in st.session_state.converted_data for msg in conv["messages"] if msg["role"] == "assistant")

            st.metric("用户消息", user_messages)
            st.metric("助手消息", ai_messages)

        # 显示对话预览
        tab1, tab2 = st.tabs(["对话预览", "完整JSON数据"])

        with tab1:
            st.subheader("对话路径预览")

            for idx, conv in enumerate(st.session_state.converted_data[:5]):  # 只显示前5个
                with st.expander(f"对话路径 #{idx + 1} (ID: {conv['path_id'][:20]}...)"):
                    for msg in conv["messages"]:
                        if msg["role"] == "system":
                            continue

                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

        with tab2:
            st.subheader("完整JSON数据")
            st.json(st.session_state.converted_data)

        # 下载按钮
        st.download_button(
            label="下载转换结果",
            data=json.dumps(st.session_state.converted_data, ensure_ascii=False, indent=2),
            file_name=f"chatml_output_{int(time.time())}.json",
            mime="application/json",
            type="primary"
        )

        # 重置按钮
        if st.button("上传新文件", type="secondary"):
            st.session_state.converted_data = None
            st.session_state.uploaded_file = None
            st.experimental_rerun()

# 示例文件下载
st.markdown("---")
st.markdown("### 示例文件")

if st.button("下载示例.canvas文件"):
    # 创建示例数据
    example_data = {
        "nodes": [
            {"id": "node1", "type": "text", "text": "用户：最近压力好大\n\nAI：我能理解你的感受，能具体说说发生了什么吗？"},
            {"id": "node2", "type": "text",
             "text": "用户：工作太多，完全做不完\n\nAI：听起来工作任务确实很重，我们来分析一下优先级？"}
        ],
        "edges": [
            {"id": "edge1", "fromNode": "node1", "toNode": "node2"}
        ]
    }

    # 创建下载链接
    st.download_button(
        label="下载示例文件",
        data=json.dumps(example_data, indent=2),
        file_name="example.canvas",
        mime="application/json"
    )

# 使用说明
st.markdown("---")
st.markdown("### 使用说明")
st.markdown("""
1. **准备.canvas文件**：确保文件包含有效的对话数据（用户和AI对话）
2. **上传文件**：通过拖放或点击选择文件区域上传
3. **转换格式**：点击"转换为ChatML格式"按钮
4. **查看结果**：预览转换后的对话或下载完整JSON数据
5. **阿里云百炼使用**：将转换后的JSON文件上传至阿里云百炼平台进行微调
""")

# 注意事项
st.warning("""
**注意**：
- 请确保在运行此应用前启动FastAPI服务（运行 `python api.py`）
- 转换服务运行在 `http://localhost:8000`
- 如需部署到生产环境，请配置合适的API地址
""")
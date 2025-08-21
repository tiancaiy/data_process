import json
import re
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
import datetime

app = FastAPI(title="Canvas to ChatML Converter API")

# 允许跨域请求 - Streamlit UI需要
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_canvas_to_chatml(canvas_data):
    """核心转换函数"""
    # 构建对话图谱
    graph = defaultdict(list)
    for edge in canvas_data["edges"]:
        graph[edge["fromNode"]].append(edge["toNode"])

    # 节点文本解析器
    def parse_node_text(text):
        user_content = re.search(r"用户：(.*?)(\n\nAI：|$)", text, re.DOTALL)
        ai_content = re.search(r"AI：(.*?)(\n\n用户：|$)", text, re.DOTALL)
        return (
            user_content.group(1).strip() if user_content else None,
            ai_content.group(1).strip() if ai_content else None
        )

    # 节点字典 {id: node}
    nodes = {node["id"]: node for node in canvas_data["nodes"]}

    # 找到根节点（假设第一个节点是根节点）
    root_id = canvas_data["nodes"][0]["id"] if canvas_data["nodes"] else None

    if not root_id:
        return {"messages": []}

    # 递归遍历所有路径
    def traverse_paths(start_id, path=None):
        if path is None:
            path = []

        # 添加当前节点
        if start_id not in path:  # 避免循环
            path.append(start_id)

            # 处理所有后继节点
            next_ids = graph.get(start_id, [])
            if not next_ids:  # 终点
                return [path]

            all_paths = []
            for next_id in next_ids:
                new_paths = traverse_paths(next_id, path.copy())
                all_paths.extend(new_paths)
            return all_paths
        return [path]

    # 从根节点开始遍历
    all_paths = traverse_paths(root_id)

    # 处理分支路径
    for from_id, to_ids in graph.items():
        if len(to_ids) > 1:  # 识别分叉点
            for to_id in to_ids:
                if to_id not in [node for path in all_paths for node in path]:
                    branch_paths = traverse_paths(to_id)
                    all_paths.extend(branch_paths)

    # 将所有路径转换为ChatML格式
    all_conversations = []

    for path in all_paths:
        messages = []

        # 添加系统提示
        messages.append({
            "role": "system",
            "content": "你是一个心理支持助手，使用认知行为疗法技术帮助用户管理焦虑情绪。"
        })

        for node_id in path:
            if node_id in nodes:
                user_text, ai_text = parse_node_text(nodes[node_id]["text"])
                if user_text:
                    messages.append({"role": "user", "content": user_text})
                if ai_text:
                    messages.append({"role": "assistant", "content": ai_text})

        if len(messages) > 1:  # 至少包含用户和助手各一条消息
            all_conversations.append({
                "messages": messages,
                "path_id": "->".join(path)
            })

    return all_conversations


@app.post("/convert")
async def convert_canvas(file: UploadFile = File(...)):
    """接收并处理上传的.canvas文件"""
    try:
        # 读取上传文件内容
        contents = await file.read()
        canvas_data = json.loads(contents)

        # 处理文件
        chatml_data = parse_canvas_to_chatml(canvas_data)

        return JSONResponse(content=chatml_data)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"文件处理失败: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
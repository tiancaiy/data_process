import json
import re
from collections import defaultdict


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

    # 从根节点开始构建主路径
    root_id = "c1366488ea397e30"
    main_path = []
    current = root_id
    while current:
        main_path.append(current)
        current = graph.get(current, [None])[0]  # 取第一个分支

    # 处理分支路径
    branch_paths = []
    for from_id, to_ids in graph.items():
        if len(to_ids) > 1:  # 识别分叉点
            for to_id in to_ids:
                path = []
                current = to_id
                while current:
                    path.append(current)
                    current = graph.get(current, [None])[0]
                branch_paths.append(path)

    # 构建完整对话链
    conversations = []
    full_path = main_path + [node for branch in branch_paths for node in branch]

    # 转换为ChatML格式
    messages = []
    for node_id in full_path:
        if node_id not in nodes:
            continue
        user_text, ai_text = parse_node_text(nodes[node_id]["text"])
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if ai_text:
            messages.append({"role": "assistant", "content": ai_text})

    return {"messages": messages}


# 示例使用
with open("data.canvas", encoding='utf-8') as f:
    canvas_data = json.load(f)

chatml_data = parse_canvas_to_chatml(canvas_data)

# 输出百炼格式
with open("output.jsonl", "w", encoding='utf-8') as f:
    f.write(json.dumps(chatml_data, ensure_ascii=False))
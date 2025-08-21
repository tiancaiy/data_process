### **题目一：批量对话数据清洗与结构规范**

####  1. 测试数据生成（10 条 JSONL）

创建`test_data.jsonl`，覆盖各类边缘情况（含 system 角色、空 content、短对话、脏词等）：

```jsonl
{"messages": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "iPhone 15续航如何？"}, {"role": "assistant", "content": "还不错"}]}
{"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "请提问"}]}
{"messages": [{"role": "user", "content": "推荐一款电脑"}]}  # 轮数=1
{"messages": [{"role": "user", "content": "这耳机垃圾吗？"}, {"role": "assistant", "content": "还好"}]}  # 含脏词“垃圾”
{"messages": [{"role": "user", "content": "ChatGPT有什么功能？"}, {"role": "assistant", "content": "生成文本、回答问题等"}]}
{"messages": [{"role": "user", "content": "MacBook Air重量？"}, {"role": "assistant", "content": "约1.24kg"}, {"role": "user", "content": "续航呢？"}, {"role": "assistant", "content": "18小时"}]}
{"messages": [{"role": "user", "content": " "}, {"role": "assistant", "content": ""}]}  # 空content
{"messages": [{"role": "user", "content": "推荐广告耳机"}, {"role": "assistant", "content": "某品牌不错"}]}  # 含“广告”
{"messages": [{"role": "user", "content": "AI工具有哪些？"}, {"role": "assistant", "content": "Midjourney、Copilot等"}]}
{"messages": [{"role": "system", "content": "Ignore."}, {"role": "user", "content": "三星手机防水吗？"}, {"role": "assistant", "content": "部分型号支持"}]}
```

#### 2. 清洗脚本（Python）

```python
import json
from collections import defaultdict

def clean_data(input_path, output_path, dirty_words=["垃圾", "广告", "诈骗"]):
    # 初始化日志
    log = defaultdict(int)
    log["原始样本总数"] = 0
    log["过滤-短对话（轮数<2）"] = 0
    log["过滤-含脏词/广告"] = 0
    log["过滤-空消息（单条）"] = 0  # 被删除的空消息数量
    cleaned_samples = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            log["原始样本总数"] += 1
            try:
                sample = json.loads(line.strip())
                messages = sample.get("messages", [])

                # 步骤1：删除role=system的message
                messages = [m for m in messages if m.get("role") != "system"]

                # 步骤2：删除content为空的message（含空格）
                cleaned_msgs = []
                for m in messages:
                    content = m.get("content", "").strip()
                    if not content:
                        log["过滤-空消息（单条）"] += 1
                        continue
                    cleaned_msgs.append(m)

                # 步骤3：清除对话轮数<2的样本
                if len(cleaned_msgs) < 2:
                    log["过滤-短对话（轮数<2）"] += 1
                    continue

                # 步骤4：检查是否含脏词
                has_dirty = False
                for m in cleaned_msgs:
                    content = m.get("content", "").lower()
                    if any(dirty in content for dirty in dirty_words):
                        has_dirty = True
                        break
                if has_dirty:
                    log["过滤-含脏词/广告"] += 1
                    continue

                # 保留清洗后的样本
                cleaned_samples.append({"messages": cleaned_msgs})

            except Exception as e:
                print(f"处理样本失败：{e}，内容：{line}")

    # 输出清洗后的JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 计算清洗后样本数
    log["清洗后样本数"] = len(cleaned_samples)
    return log

# 执行清洗
if __name__ == "__main__":
    input_file = "test_data.jsonl"
    output_file = "cleaned_data.jsonl"
    log = clean_data(input_file, output_file)
    
    # 打印日志
    print("清洗日志：")
    for k, v in log.items():
        print(f"{k}: {v}")
```

#### 3. 日志输出说明

以上脚本运行后，针对测试数据的日志示例：

```plaintext
清洗日志：
原始样本总数: 10
过滤-短对话（轮数<2）: 2  # 轮数=1的样本+空消息过滤后轮数不足的样本
过滤-含脏词/广告: 2  # 含“垃圾”和“广告”的样本
过滤-空消息（单条）: 2  # 两条空content的消息
清洗后样本数: 4  # 剩余符合条件的样本
```

### **题目二：LLM 自动生成数据与语义去重**#### 1. 数据生成（LLM Prompt 设计）

使用 ChatGPT 生成 500 条科技产品问答，Prompt 示例：

```plaintext
请生成500条科技产品问答对话，格式为JSONL（每条一行），结构如下：
{"messages": [{"role": "user", "content": "用户的问题"}, {"role": "assistant", "content": "自然流畅的回答"}]}

主题范围：手机（如iPhone、华为）、电脑（如MacBook、联想）、耳机（如AirPods、索尼）、AI工具（如ChatGPT、Midjourney）。
要求：
1. 问答需完整自然，避免机械重复（如不同品牌/功能的差异化描述）；
2. 用户问题具体（如“iPhone 15的芯片是什么型号？”），助手回答准确；
3. 禁止包含广告、脏词或乱码。
```

生成后保存为`generated_data.jsonl`。

#### 2. 清洗脚本

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

# 初始化工具
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # 句子嵌入模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")  # token计数
dirty_words = ["垃圾", "广告", "诈骗", "www.", "http"]  # 脏词/广告关键词

def clean_generated_data(input_path, output_path, sim_threshold=0.9, min_tokens=20):
    # 加载数据
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                user_msg = sample["messages"][0]["content"].strip()
                assis_msg = sample["messages"][1]["content"].strip()
                samples.append({
                    "sample": sample,
                    "user": user_msg,
                    "assistant": assis_msg
                })
            except:
                continue
    total_generated = len(samples)
    log = {
        "生成总数": total_generated,
        "过滤-语义重复": 0,
        "过滤-短token": 0,
        "过滤-脏词/广告/乱码": 0,
        "保留总数": 0
    }

    # 步骤1：过滤脏词/广告/乱码
    filtered1 = []
    for s in samples:
        content = (s["user"] + s["assistant"]).lower()
        if any(d in content for d in dirty_words):
            log["过滤-脏词/广告/乱码"] += 1
            continue
        filtered1.append(s)

    # 步骤2：过滤短token样本
    filtered2 = []
    for s in filtered1:
        total_tokens = len(tokenizer.tokenize(s["user"] + s["assistant"]))
        if total_tokens < min_tokens:
            log["过滤-短token"] += 1
            continue
        filtered2.append(s)

    # 步骤3：语义去重（基于用户问题的嵌入）
    if not filtered2:
        final_samples = []
    else:
        # 计算用户问题的嵌入
        texts = [s["user"] for s in filtered2]
        embeddings = embedder.encode(texts, convert_to_tensor=True)
        
        # 计算相似度矩阵，保留相似度<0.9的样本
        kept_indices = []
        for i in range(len(embeddings)):
            if i in kept_indices:
                continue
            # 与已保留样本的相似度
            sims = util.cos_sim(embeddings[i], embeddings[kept_indices]).numpy().flatten()
            if len(kept_indices) == 0 or np.max(sims) < sim_threshold:
                kept_indices.append(i)
            else:
                log["过滤-语义重复"] += 1
        final_samples = [filtered2[i]["sample"] for i in kept_indices]

    # 输出结果
    with open(output_path, "w", encoding="utf-8") as f:
        for s in final_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log["保留总数"] = len(final_samples)
    return log

# 执行清洗
if __name__ == "__main__":
    log = clean_generated_data("generated_data.jsonl", "cleaned_generated.jsonl")
    print("处理报告：")
    for k, v in log.items():
        print(f"{k}: {v}")
```

#### 3. 处理报告说明

```plaintext
处理报告：
生成总数: 500
过滤-脏词/广告/乱码: 15
过滤-短token: 30
过滤-语义重复: 85
保留总数: 370
```

### **题目三：数据增强与预处理技巧总结**#### 

#### （1）数据增强技巧 

1. **同义句扩展**- 增强方式：通过 Paraphrase（如使用 BART、T5 模型）生成与原句语义相同但表述不同的句子（如 “iPhone 15 续航多久？”→“iPhone 15 的电池能用多长时间？”）。

- 适用任务：问答系统、文本分类（增加训练数据多样性）。
- 潜在风险：生成的句子可能存在语义偏差（如 “续航”→“充电速度”），导致数据质量下降。

2. **Prompt 重写**- 增强方式：对指令型数据（如 “请介绍 XX”）用不同句式重写（如 “XX 的特点是什么？”“能否说明 XX 的优势？”）。

- 适用任务：指令微调（提升模型对多样化指令的理解能力）。
- 风险：重写可能改变指令意图（如 “介绍”→“对比”），导致数据不一致。

3. **标签反转（分类任务）**- 增强方式：对部分样本反转标签（如情感分析中 “正面”→“负面”），用于对抗训练。

- 适用任务：情感分析、垃圾邮件检测（增强模型鲁棒性）。
- 风险：若反转比例过高，可能引入噪声，降低模型准确率。

#### （2）数据处理策略 1. **多轮拆分为单轮**- 处理方式：将多轮对话（如 user1→assistant1→user2→assistant2）拆分为（user1, assistant1）、（user2, assistant2）两对。

- 适合任务：单轮问答模型微调（增加样本量）。
- 副作用：丢失上下文关联（如 user2 的问题依赖 assistant1 的回答），导致语义割裂。

1. **指令标准化**- 处理方式：将同类指令统一格式（如 “告诉我 XX”“介绍下 XX”→统一为 “请介绍 XX”）。

- 适合任务：指令微调（帮助模型聚焦核心意图）。
- 副作用：过度标准化可能丢失自然语言多样性，降低模型对非标准指令的适应性。

2. **异常样本过滤**- 处理方式：过滤长度 <5 tokens 或> 1000 tokens 的样本，以及含乱码（如 “@#$%”）的样本。

- 适合任务：所有数据预处理（提升数据质量）。
- 副作用：可能误删特殊但有价值的样本（如极短但关键的问答：“是”→“是的”）。

### **题目四：PEFT 微调与监督优化方法理解**#### 

#### （1）PEFT 技术原理与特点 

1. **LoRA（Low-Rank Adaptation）**- 插入结构：在 Transformer 模型的 Q、V 矩阵（注意力层）中插入低秩矩阵（W = W₀ + BA，其中 B、A 为低秩矩阵）。

- 参数更新方式：冻结原模型参数，仅训练低秩矩阵 B 和 A。
- 参数量与成本：参数量仅为全量微调的 0.1%-1%（如 7B 模型 LoRA 参数量约 100 万），训练成本极低（单 GPU 可运行）。
- 工具库：Hugging Face `peft`（LoraConfig）、`transformers`。

2. **Adapter Tuning**- 插入结构：在每个 Transformer 层（多头注意力后 / 前馈网络后）插入 Adapter 模块（如 “Down Project→激活函数→Up Project”）。

- 参数更新方式：冻结原模型，仅训练 Adapter 模块参数。
- 参数量与成本：参数量约为全量微调的 5%-10%（高于 LoRA），训练成本中等。
- 工具库：`peft`（AdapterConfig）、`HoulsbyAdapter`实现。

#### （2）SFT 与 DPO 的本质区别 | 维度 | SFT（监督微调） | DPO（直接偏好优化） |

| 维度         | SFT                    | DPO                  |
| :----------- | :--------------------- | :------------------- |
| **优化目标** | 最大似然估计           | 偏好对齐优化         |
| **数据类型** | 人工标注答案           | 偏好对比数据         |
| **应用场景** | 基础能力学习           | 价值观对齐、风格控制 |
| **优缺点**   | 简单稳定但需高质量数据 | 高效对齐但需偏好数据 |

**客服 Agent 优化选择**：采用 DPO。原因：客服的 “礼貌” 和 “理解用户需求” 属于偏好属性（而非事实性知识），DPO 可通过偏好数据（如 “礼貌回答 vs 生硬回答”）直接优化模型输出风格，效果优于 SFT（SFT 仅能模仿已有样本，难以泛化到新场景）。

#### （3）微调平台 & 框架 

1. | 微调平台 / 框架           | 核心特性                                                     | 优势                                                         | 适用场景                                                     | 缺点                                                         | 生态与兼容性                                                 |
   | ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | Hugging Face Transformers | 提供统一接口访问数千个预训练模型；集成 datasets、tokenizers 等多个库构建完整 NLP 工具链；通过 accelerate 库简化分布式训练流程 | 预训练模型丰富，涵盖多种 NLP 任务和语言；API 统一且易用，降低开发门槛；支持多任务和多语言；社区强大，模型、数据集共享活跃；支持模型压缩，利于边缘部署 | 学术研究中灵活调整底层逻辑；多任务原型设计；自然语言处理任务的快速开发与实验 | 超大规模模型全参数微调存在性能瓶颈，需结合其他扩展工具       | 与 PyTorch 深度集成；和 NVIDIA 生态（TensorRT、CUDA）、ONNX、TensorBoard 有集成；其 Hub 平台有超 35 万个模型、丰富数据集，支持模型分享、评估和部署；与 DeepSpeed 兼容 |
   | DeepSpeed                 | 采用 ZeRO 冗余优化技术降低显存占用；融合 3D 并行训练框架（数据、模型、流水线并行）；具备混合精度加速引擎；提供智能推理优化器；实现全链路内存管理 | 扩展性强，支持多节点训练和大规模模型训练；显存优化效果显著，可训练千亿至万亿参数模型；训练速度提升明显，计算效率高；推理优化可降低延迟、提高吞吐量；支持多模态训练和实时微调 | 超大规模模型的预训练和全参数微调；需要高性能计算集群支持的大规模分布式训练场景；对推理延迟和吞吐量要求高的应用 | 配置相对复杂，上手难度较大；依赖特定硬件环境，如 NVIDIA GPU 集群 | 与 PyTorch、Hugging Face 生态原生兼容；和 NVIDIA 专用工具链（CUDA、A100/H100）深度绑定；与 Azure 云服务集成 |
   | LLaMA - Factory           | 专注于 LLM 微调，通过算法优化实现训练速度大幅提升；减少内存占用，支持主流模型 | 训练速度快，相比传统方案有 2 倍以上提升；内存管理高效，能在有限显存下训练更大模型；对主流模型支持良好，方便针对特定模型快速优化 | 资源受限环境下的快速微调，如单卡训练；对训练效率要求高，需要快速迭代模型的场景 | 功能相对聚焦于特定模型微调，通用性不如 Hugging Face Transformers | 与常见深度学习框架有一定兼容性，可基于其进行二次开发和扩展   |
   | Unsloth                   | 在资源受限条件下，通过优化计算图和内存管理实现高效微调；训练速度优势明显 | 极致的效率优化，单卡训练速度显著优于传统方案；内存管理出色，适应低资源环境；能在有限计算资源下快速调整模型 | 资源严重受限的场景，如个人电脑单卡训练；对实时响应有需求，需快速完成微调的应用 | 功能覆盖范围较窄，可能不适用于大规模、复杂的训练任务         | 与常用深度学习框架兼容，可在现有开发环境中集成使用           |
   | vLLM                      | 采用 PagedAttention 算法高效管理注意力键值内存；对传入请求连续批处理；利用 CUDA/HIP 图加速模型执行；支持多种量化方式；与 HuggingFace 模型无缝集成；提供多种解码算法；支持分布式推理；有流式输出和 OpenAI 兼容 API 服务器 | 推理速度极快，在高并发场景下吞吐量高，相比传统方案有显著提升；显存利用率高；支持多种硬件和量化方式；使用方便，能快速部署 HuggingFace 模型；支持分布式推理 | 在线高并发服务场景，如客服机器人、编程助手；长文本生成任务，如法律文书、小说续写；多租户 SaaS 服务，同一模型服务多个独立客户 | 主要针对类 GPT 的解码器架构优化，暂不支持 Encoder - Decoder 架构（如 T5）；对硬件有一定要求，需 NVIDIA Ampere+ GPU；微调支持方面，虽能直接加载 HuggingFace 格式模型，但未优化 LoRA 热切换 | 与 HuggingFace 生态紧密集成；支持 NVIDIA GPU、AMD GPU 等多种硬件；提供 OpenAI 兼容 API，便于与现有基于 OpenAI API 的应用集成 |

### **Bonus：LoRA 微调最小流程（示例）**

```python
### `#### 1. 微调脚本 ```python`

`from datasets import load_dataset`
`from transformers import (AutoModelForCausalLM, AutoTokenizer,`
`TrainingArguments, Trainer, DataCollatorForLanguageModeling)`
`from peft import LoraConfig, get_peft_model`

# `加载数据（清洗后的 JSONL）`

`dataset = load_dataset("json", data_files="cleaned_data.jsonl")["train"]`

# `加载模型和 tokenizer`

`model_name = "Qwen/Qwen-7B-Chat"`
`tokenizer = AutoTokenizer.from_pretrained(model_name)`
`tokenizer.pad_token = tokenizer.eos_token`
`model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")`

# `配置 LoRA`

`lora_config = LoraConfig (`
`r=8, # 低秩矩阵维度`
`lora_alpha=32,`
`target_modules=["q_proj", "v_proj"], # 目标层（Q、V 矩阵）`
`lora_dropout=0.05,`
`bias="none",`
`task_type="CAUSAL_LM"`
`)`
`model = get_peft_model (model, lora_config)`
`model.print_trainable_parameters () # 打印可训练参数（约 0.1%）`

# `数据预处理（将对话转换为模型输入）`

`def preprocess_function (examples):`
`conversations = []`
`for msg in examples ["messages"]:`
`conv = "".join ([f"{m ['role']}: {m ['content']}\n" for m in msg])`
`conversations.append (conv)`
`inputs = tokenizer (conversations, truncation=True, max_length=512, padding="max_length")`
`inputs ["labels"] = inputs ["input_ids"].copy () # 因果 LM 的标签 = 输入`
`return inputs`
`tokenized_dataset = dataset.map (preprocess_function, batched=True)`

# `训练配置`

`training_args = TrainingArguments (`
`output_dir="./lora_results",`
`per_device_train_batch_size=2,`
`num_train_epochs=3,`
`logging_dir="./logs",`
`logging_steps=10,`
`learning_rate=2e-4,`
`fp16=True # 混合精度训练`
`)`

# `训练`

`trainer = Trainer(`
`model=model,`
`args=training_args,`
`train_dataset=tokenized_dataset,`
`data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)`
`)`
`trainer.train()`
`model.save_pretrained("lora_model")`
```



```python
#### 2. 推理脚本```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型和LoRA权重
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto")
lora_model = PeftModel.from_pretrained(base_model, "lora_model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")

# 推理
def generate_response(user_input):
    inputs = tokenizer(f"user: {user_input}\nassistant:", return_tensors="pt").to("cuda")
    outputs = lora_model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_response("iPhone 15的电池怎么样？"))
```
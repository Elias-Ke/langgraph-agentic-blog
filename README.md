# 大模型工程化实战：从 RAG 到 Agentic RAG

系统学习 LLM 的代码与教程仓库。每篇文章对应一个独立的可运行代码目录。

## 环境准备

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) 包管理器

```bash
# 1. 安装依赖
uv sync

# 2. 配置 API Key（Stage 2 需要，Stage 1 不需要）
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
```

## Stage 1：Embedding 与 Attention 机制


本阶段不依赖 API，纯本地计算。

```bash
# 第一章：Embedding 本质
uv run python stage1-embedding-attention/week1-attention/embedding_demo.py       # 实验1+2+3: 相似度/语义类比/t-SNE

# 第二章：Tokenization
uv run python stage1-embedding-attention/week1-attention/token_counter.py        # tiktoken 中/英/代码 Token 消耗对比

# 第三章：RNN/LSTM
uv run python stage1-embedding-attention/week1-attention/rnn_lstm_demo.py        # RNN → LSTM → GRU
uv run python stage1-embedding-attention/week1-attention/bahdanau_attention.py   # Bahdanau Attention

# 第四章：Self-Attention
uv run python stage1-embedding-attention/week1-attention/self_attention_numpy.py # NumPy 手写 Self-Attention
uv run python stage1-embedding-attention/week1-attention/manual_attention.py     # PyTorch 手写 + 带权重的 SelfAttention 类

# 第五章：多头注意力 + 位置编码
uv run python stage1-embedding-attention/week1-attention/multi_head_attention.py # 多头注意力 (NumPy)
uv run python stage1-embedding-attention/week1-attention/positional_encoding.py  # 正弦/余弦位置编码
```

## Stage 2：LangChain 极简入门

> 📄 文章：`stage2-langchain-intro/article.md`

本阶段需要调用 LLM API，请先完成环境准备中的 `.env` 配置。

```bash
uv run python stage2-langchain-intro/hello_llm.py         # 示例 1: Chat Model 调用
uv run python stage2-langchain-intro/prompt_template.py   # 示例 2: ChatPromptTemplate
uv run python stage2-langchain-intro/simple_chain.py      # 示例 3: LCEL 链式调用
```

## 文章与代码对照

| 序号 | 文章 | 代码目录 | 依赖 API |
|------|------|---------|---------|
| 1 | 深度学习教程：Embedding 与 Attention 机制 | `stage1-embedding-attention/week1-attention/` | 否 |
| 2 | LangChain 极简入门 | `stage2-langchain-intro/` | 是 |

## 切换模型

编辑 `.env` 中的 `DEEPSEEK_MODEL` 和 `DEEPSEEK_BASE_URL` 即可切换任意 OpenAI 兼容接口的模型。

## 目录结构

```
├── .env.example
├── pyproject.toml
├── 深度学习教程_Embedding与Attention机制.md   # 文章1
├── stage1-embedding-attention/
│   └── week1-attention/
│       ├── embedding_demo.py                 # Ch1.4: 向量相似度 + 语义类比 + t-SNE
│       ├── token_counter.py                  # Ch2.5: Token 消耗对比
│       ├── test_zh.txt                       #   测试语料（中文，685 字）
│       ├── test_en.txt                       #   测试语料（英文，2205 字符）
│       ├── rnn_lstm_demo.py                  # Ch3.1-3.3: RNN/LSTM/GRU
│       ├── bahdanau_attention.py             # Ch3.5: Bahdanau Attention
│       ├── self_attention_numpy.py           # Ch4.5: NumPy 手写 Self-Attention
│       ├── manual_attention.py               # Ch4.6: PyTorch SelfAttention
│       ├── multi_head_attention.py           # Ch5.1: 多头注意力
│       └── positional_encoding.py            # Ch5.2: 正弦/余弦位置编码
└── stage2-langchain-intro/
    ├── article.md
    ├── README.md
    ├── llm_config.py
    ├── hello_llm.py
    ├── prompt_template.py
    └── simple_chain.py
```

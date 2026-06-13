# Stage 1：Embedding 与 Attention 机制

本阶段不依赖 API，纯本地计算。所有实验在 2024 年 MacBook Pro 上 < 60 秒完成。

## 运行

```bash
# 第一章：Embedding 本质 — 语义相似度、向量运算、t-SNE 可视化
uv run python stage1-embedding-attention/week1-attention/embedding_demo.py

# 第二章：Tokenization — 6 个主流模型分词器中/英/代码 Token 消耗对比
uv run python stage1-embedding-attention/week1-attention/token_counter.py

# 第三章：RNN/LSTM — 序列建模从 RNN 到 LSTM 到 Attention
uv run python stage1-embedding-attention/week1-attention/rnn_lstm_demo.py        # RNN → LSTM → GRU 结构对比
uv run python stage1-embedding-attention/week1-attention/bahdanau_attention.py   # Bahdanau Attention 实现

# 第四章：Self-Attention — 从零实现缩放点积注意力
uv run python stage1-embedding-attention/week1-attention/self_attention_numpy.py # NumPy 手写，打印每步中间结果
uv run python stage1-embedding-attention/week1-attention/manual_attention.py     # PyTorch 完整 SelfAttention 类

# 第五章：多头注意力 + 位置编码
uv run python stage1-embedding-attention/week1-attention/multi_head_attention.py # 多头分拆 + Concat 拼接
uv run python stage1-embedding-attention/week1-attention/positional_encoding.py  # 正弦/余弦波形可视化
```

## Demo 详解

| Demo | 做什么 | 关键输出 |
|------|--------|---------|
| `embedding_demo.py` | 用 BGE 中文模型编码句子，计算余弦相似度，验证 king−man+woman≈queen，t-SNE 可视化 | 相似度矩阵、向量运算结果、聚类散点图 |
| `token_counter.py` | 用 tiktoken + HuggingFace 6 个分词器对比中/英/代码 Token 消耗 | 685 字技术文章的 Token 对比表 |
| `rnn_lstm_demo.py` | 对比 RNN / LSTM / GRU 的结构差异和输出形状 | 三种模型的参数统计和输出维度 |
| `bahdanau_attention.py` | 实现 Bahdanau 加法注意力，展示 Q/K/V 维度流转 | Attention 权重矩阵（每行和为 1） |
| `self_attention_numpy.py` | 用 NumPy 手写 Scaled Dot-Product Attention，打印每一步 | Q·K^T 分数矩阵 → 缩放 → Softmax → 加权求和 |
| `manual_attention.py` | PyTorch 版 SelfAttention 类，带权重 + Mask 支持 | 可训练的参数矩阵 W_q/W_k/W_v |
| `multi_head_attention.py` | 多头注意力：Split → 并行计算 → Concat → W_o 投影 | 2 头 × 4 维 → 拼接回 8 维的全过程 |
| `positional_encoding.py` | 正弦/余弦位置编码的生成和波形可视化 | 4 个维度的正弦波曲线图 |

# 系统学习 Agent — 代码与教程仓库

系统学习 AI Agent 的代码与教程仓库。每篇文章对应一个独立的可运行代码目录。

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 配置 API Key（Stage 2 需要，Stage 1 不需要）
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

## 项目结构

```
├── stage1-embedding-attention/      # 第一章：Embedding 与 Attention → README
│   └── week1-attention/             # 8 个 Demo，纯本地计算，不依赖 API
└── stage2-prompt-engineering/       # 第二章：提示工程 + LangChain → README
    ├── exp1 ~ exp5                  # 上篇：纯 OpenAI SDK
    └── exp6 ~ exp8                  # 下篇：LangChain
```

每个子项目的详细说明见各自的 `README.md`：

- [`stage1-embedding-attention/README.md`](stage1-embedding-attention/README.md) — 8 个 Demo：Embedding → Tokenization → RNN → Attention → Self-Attention → 多头 → 位置编码
- [`stage2-prompt-engineering/README.md`](stage2-prompt-engineering/README.md) — 8 个 Demo：Zero-shot → Few-shot → CoT → ReAct → 系统提示词 → 结构化输出 → LangChain 三核心 → @tool 绑定

## 文章与代码对照

| 文章 | 代码 | 依赖 API |
|------|------|---------|
| 深度学习教程：Embedding 与 Attention 机制 | `stage1-embedding-attention/` | 否 |
| 提示工程核心教程（上篇） | `stage2-prompt-engineering/exp1~5/` | 是 |
| LangChain 极简入门（下篇） | `stage2-prompt-engineering/exp6~8/` | 是 |

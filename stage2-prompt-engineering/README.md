# Stage 2：提示工程 + LangChain 入门

本阶段需要调用 LLM API。先配置 `.env`：

```bash
# 在项目根目录
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

所有实验使用 `deepseek-v4-flash` 模型。

---

## 上篇：提示工程核心教程（纯 OpenAI SDK）

上篇 5 个实验不依赖 LangChain，纯 `openai` SDK + Python 手写。

| Demo | 命令 | 做什么 | 关键输出 |
|------|------|--------|---------|
| `exp1` | `uv run python stage2-prompt-engineering/exp1-zeroshot-fewshot/classify.py` | Zero-shot vs Few-shot 分类对比。3 个难度任务 × 3 种 shot 模式，对比准确率 | 情感/意图/逻辑分类 0-shot vs 1-shot vs 3-shot 准确率表 |
| `exp2` | `uv run python stage2-prompt-engineering/exp2-cot/math_reasoning.py` | CoT 思维链对比。同一个数学题用 3 种方式问 | Zero-shot / Zero-shot CoT / Few-shot CoT 的推理路径和答案对比 |
| `exp3` | `uv run python stage2-prompt-engineering/exp3-react/react_loop.py` | 手写 ReAct Agent 循环。30 行 while 循环 + 2 个工具 | 3 个递进问题的 Thought→Action→Observation 完整执行轨迹 |
| `exp4` | `uv run python stage2-prompt-engineering/exp4-system-prompt/code_reviewer.py` | 系统提示词架构化。对比「随便写的」vs「五层架构」的代码审查质量 | 两种提示词对同一段代码的审查报告对比 |
| `exp5` | `uv run python stage2-prompt-engineering/exp5-structured-output/entity_extract.py` | 结构化输出。对比 JSON Mode vs JSON Schema strict 模式 | 实体关系提取的 JSON 输出 + Schema 合规检查 |

## 下篇：LangChain 快速入门

下篇 3 个实验用 LangChain 重构上篇的逻辑，展示框架帮你省了什么。

| Demo | 命令 | 做什么 | 关键输出 |
|------|------|--------|---------|
| `exp6` | `uv run python stage2-prompt-engineering/exp6-langchain-basics/core_concepts.py` | LangChain 三个核心概念：ChatModel + PromptTemplate + 管道 `\|` | Few-shot 模板渲染结果 + 链式调用输出 |
| `exp7` | `uv run python stage2-prompt-engineering/exp7-langchain-modes/four_modes.py` | LangChain 版 Zero-shot → ReAct 四种模式 | 手写版 vs LangChain 版的代码量对比 |
| `exp8` | `uv run python stage2-prompt-engineering/exp8-langchain-tools/tool_binding.py` | `@tool` 装饰器 + `bind_tools`。LLM 自主决定是否调用工具 | bind_tools 后 LLM 的 tool_calls |

## 快速体验

从最核心的三个实验开始：

```bash
# 1. 手写 ReAct Agent（最重要的 Demo）
uv run python stage2-prompt-engineering/exp3-react/react_loop.py

# 2. LangChain 三个核心概念
uv run python stage2-prompt-engineering/exp6-langchain-basics/core_concepts.py

# 3. 工具绑定 —— LLM 自主决定调用哪个工具
uv run python stage2-prompt-engineering/exp8-langchain-tools/tool_binding.py
```

## 切换模型

所有实验默认使用 `deepseek-v4-flash`。换模型时修改脚本中的 `model="deepseek-v4-flash"` 为目标模型名，同时确保 `.env` 中的 `DEEPSEEK_BASE_URL` 指向对应 API 端点。

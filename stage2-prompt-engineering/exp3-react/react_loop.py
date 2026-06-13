"""实验3：手写 ReAct 循环

不使用 LangChain，纯 Python + OpenAI SDK 实现
Thought → Action → Observation 循环。

这是理解 Agent 工作机制的最佳方式——去掉框架魔法，
看清 Agent 的底层就是一个 while 循环 + 工具调用。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from tools import search, calculator

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

MAX_ITERATIONS = 5

SYSTEM_PROMPT = """你是一个能够使用工具的智能助手。

你可以使用以下工具：
1. search(query) — 搜索信息。用于查找事实、数据、人物信息。
   例如：search("特斯拉2024年营收")
2. calculator(expression) — 执行数学计算。
   例如：calculator("977 / 55")

响应格式：
```
Thought: <分析当前状况，决定下一步做什么>
Action: <工具名称>
Action Input: <工具参数>
```

或者当你已有足够信息回答时：
```
Thought: <我已经有足够信息来回答>
Final Answer: <最终答案>
```

每次只能执行一个 Action。收到 Observation 后，再决定下一步。"""


def run_react_agent(question: str) -> list[dict]:
    """运行 ReAct 循环，返回完整的执行轨迹。

    Args:
        question: 用户问题

    Returns:
        执行轨迹列表，每步含: iteration, thought, action,
        action_input, observation, final_answer
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    trace = []

    for i in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=0.0,
        )
        text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": text})

        # 解析 Thought / Action / Action Input / Final Answer
        thought = _extract_field(text, "Thought")
        action = _extract_field(text, "Action")
        action_input = _extract_field(text, "Action Input")
        final_answer = _extract_field(text, "Final Answer")

        step = {
            "iteration": i + 1,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": None,
            "final_answer": final_answer,
        }

        # 如果有 Final Answer，结束循环
        if final_answer:
            trace.append(step)
            break

        # 执行工具调用
        if action == "search":
            obs = search(action_input)
        elif action == "calculator":
            obs = calculator(action_input)
        elif action:
            obs = f"未知工具：{action}（可用工具：search, calculator）"
        else:
            obs = "错误：未指定 Action。请输出 Action 和 Action Input。"

        step["observation"] = obs
        trace.append(step)

        # 将 Observation 注入对话
        messages.append({"role": "user", "content": f"Observation: {obs}"})

    return trace


def _extract_field(text: str, field: str) -> str | None:
    """从模型输出中提取指定字段"""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith(f"{field}:"):
            return line[len(f"{field}:"):].strip()
    return None


def print_trace(trace: list[dict], question: str):
    """格式化打印执行轨迹"""
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"{'='*60}")

    for step in trace:
        print(f"\n--- Step {step['iteration']} ---")

        thought = step["thought"] or "(模型未输出 Thought)"
        print(f"💭 Thought: {thought}")

        if step["action"]:
            print(f"🔧 Action: {step['action']}({step['action_input']})")

        if step["observation"]:
            obs = step["observation"]
            if len(obs) > 120:
                obs = obs[:120] + "..."
            print(f"👁️  Observation: {obs}")

        if step["final_answer"]:
            print(f"\n✅ Final Answer: {step['final_answer']}")

    total_steps = len(trace)
    print(f"\n📊 总步数: {total_steps}")
    if total_steps >= MAX_ITERATIONS and not trace[-1].get("final_answer"):
        print("⚠️  达到最大迭代次数但未给出最终答案")


if __name__ == "__main__":
    # 三个不同难度的问题
    questions = [
        # 简单：单步搜索即可
        "马斯克今年多少岁？",

        # 中等：需要搜索 + 计算
        "特斯拉2024年营收除以马斯克的年龄，结果是多少？",

        # 较难：需要多步推理
        "如果SpaceX 2024年每次发射的运营成本是2800万美元，"
        "那么2024年SpaceX的总发射成本是多少亿美元？"
        "（提示：需要先搜索SpaceX 2024年发射次数）",
    ]

    for q in questions:
        trace = run_react_agent(q)
        print_trace(trace, q)

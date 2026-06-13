"""ReAct Agent —— LangChain 版

和上篇 exp3 完全相同的逻辑，但用 @tool 管理工具 Schema、用 ChatOpenAI 替代 OpenAI client。
对比上篇手写版，看 LangChain 帮你省了什么。
"""

import os
import math
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

MAX_ITERATIONS = 5


# ===== 工具定义：@tool 替代手写 dict =====
@tool
def search(query: str) -> str:
    """搜索信息。当需要查找事实、数据或人物信息时使用。"""
    knowledge = {
        "特斯拉": "特斯拉公司（Tesla, Inc.）2024年营收为977亿美元。",
        "马斯克": "埃隆·马斯克（Elon Musk）出生于1971年6月28日，2026年55岁。",
        "SpaceX": "SpaceX 2024年完成了134次轨道发射。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"未找到关于「{query}」的信息"


@tool
def calculator(expression: str) -> str:
    """执行数学计算。输入数学表达式，返回计算结果。"""
    allowed = set("0123456789+-*/().%^ ")
    if not all(c in allowed for c in expression):
        return "错误：只支持基本数学运算"
    try:
        expr = expression.replace("^", "**")
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        return str(round(result, 4) if isinstance(result, float) else result)
    except Exception as e:
        return f"计算错误：{e}"


TOOLS = {"search": search, "calculator": calculator}


# ===== System prompt（和上篇完全一样）=====
SYSTEM_PROMPT = """你是一个能够使用工具的智能助手。

可用工具：
1. search(query) — 搜索信息。用于查找事实、数据、人物信息。
2. calculator(expression) — 执行数学计算。

响应格式：
```
Thought: <分析当前状况>
Action: <工具名称>
Action Input: <工具参数>
```
或：
```
Thought: <我已足够信息>
Final Answer: <最终答案>
```

每次只执行一个 Action。"""  # noqa: F811


def run_react_agent(question: str):
    """LangChain 版 ReAct 循环"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for i in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        content = response.content

        # 解析 Thought / Action / Action Input / Final Answer
        thought = action = action_input = final_answer = None
        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                action_input = line[13:].strip()
            elif line.startswith("Final Answer:"):
                final_answer = line[13:].strip()

        step = f"--- Step {i + 1} ---\n💭 Thought: {thought or '(无)'}"

        if final_answer:
            step += f"\n✅ Final Answer: {final_answer}"
            print(step)
            return

        messages.append(response)

        if action and action in TOOLS:
            obs = TOOLS[action].invoke(action_input or "")
            step += f"\n🔧 Action: {action}({action_input})\n👁️ Observation: {obs[:100]}"
            # 用纯文本注入 observation（和上篇 exp3 一样）
            messages.append(HumanMessage(content=f"Observation: {obs}"))
        else:
            step += f"\n⚠️ 未知工具: {action}"

        print(step)

    print("⚠️ 达到最大迭代次数，未给出最终答案")


if __name__ == "__main__":
    questions = [
        "马斯克今年多少岁？",
        "特斯拉2024年营收除以马斯克的年龄，结果是多少？",
        "如果SpaceX 2024年每次发射的运营成本是2800万美元，总发射成本是多少亿美元？",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"{'='*60}")
        run_react_agent(q)

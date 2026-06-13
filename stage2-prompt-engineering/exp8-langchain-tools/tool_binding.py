"""实验8：LangChain 工具定义与绑定

展示 @tool 装饰器 + bind_tools + ToolNode 三种方式。
"""

import os
import math
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ===== 1. @tool 装饰器 =====
@tool
def search(query: str) -> str:
    """搜索信息。当需要查找事实、数据或人物信息时使用。"""
    knowledge = {
        "特斯拉": "特斯拉2024年营收为977亿美元。CEO埃隆·马斯克出生于1971年。",
        "马斯克": "埃隆·马斯克出生于1971年6月28日，2026年55岁。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"未找到关于「{query}」的信息"


@tool
def calculator(expression: str) -> str:
    """执行数学计算。输入一个数学表达式，返回计算结果。"""
    allowed = set("0123456789+-*/().%^ ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不允许的字符"
    try:
        expr = expression.replace("^", "**")
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        return str(round(result, 4) if isinstance(result, float) else result)
    except Exception as e:
        return f"计算错误：{e}"


# ===== 2. bind_tools — 让 LLM 自己决定是否调用工具 =====
print("=" * 50)
print("方式1：bind_tools —— LLM 自主决定调用哪个工具")
print("=" * 50)

llm_with_tools = llm.bind_tools([search, calculator])

# 问题1：不需要工具
response = llm_with_tools.invoke("你好，今天天气怎么样？")
print(f"\n问题1（不需要工具）:")
print(f"  content: {response.content}")
print(f"  tool_calls: {response.tool_calls}")

# 问题2：需要工具
response = llm_with_tools.invoke("特斯拉2024年营收是多少？")
print(f"\n问题2（需要工具 search）:")
print(f"  content: {response.content}")
print(f"  tool_calls: {response.tool_calls[0]['name'] if response.tool_calls else '无'}")
if response.tool_calls:
    tc = response.tool_calls[0]
    print(f"  args: {tc['args']}")


# ===== 3. 手动执行工具调用 =====
print("\n" + "=" * 50)
print("方式2：手动执行工具调用（模拟 Agent 循环）")
print("=" * 50)

messages = [HumanMessage(content="马斯克今年多少岁？")]
response = llm_with_tools.invoke(messages)
messages.append(response)

if response.tool_calls:
    tc = response.tool_calls[0]
    tool_name = tc["name"]
    tool_args = tc["args"]

    print(f"LLM 决定调用: {tool_name}({tool_args})")

    # 执行工具
    if tool_name == "search":
        result = search.invoke(tool_args)
    elif tool_name == "calculator":
        result = calculator.invoke(tool_args)
    else:
        result = f"未知工具: {tool_name}"

    print(f"工具返回: {result[:100]}...")

    # 将结果注入对话
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # 再次调用 LLM
    final_response = llm_with_tools.invoke(messages)
    print(f"\n最终回答: {final_response.content}")


# ===== 4. 对比：手写工具 vs LangChain 工具 =====
print("\n" + "=" * 50)
print("对比：手写 dict vs LangChain @tool")
print("=" * 50)

# 手写方式（上篇 ReAct 实验）
handwritten_tool = {
    "name": "search",
    "description": "搜索信息",
    "parameters": {"query": "str"},
}

# LangChain @tool
print(f"手写工具定义: {handwritten_tool}")
print(f"LangChain @tool:")
print(f"  name: {search.name}")
print(f"  description: {search.description}")
print(f"  args_schema: {search.args_schema.model_json_schema()}")
print(f"\n优势：@tool 自动从函数签名和 docstring 生成 Schema，" +
      "不需要手动维护 JSON——改了函数签名，Schema 自动更新")

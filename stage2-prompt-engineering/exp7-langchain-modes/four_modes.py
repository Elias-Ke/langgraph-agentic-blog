"""实验7：用 LangChain 实现 Zero-shot → ReAct 四种模式

对比上篇手写版，展示 LangChain 如何减少样板代码。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

SENTENCES = [
    "这个产品质量太差了，用了两天就坏了",
    "性价比很高，推荐购买",
    "一般般吧，没什么特别的",
]
QUESTION = "父亲年龄是儿子的3倍，5年后是儿子的2倍，儿子今年几岁？"


def mode1_zeroshot():
    """模式1：Zero-shot —— LangChain 版"""
    print("\n--- 模式1：Zero-shot ---")
    template = ChatPromptTemplate.from_messages([
        ("system", "将文本分类为：正面/负面/中性。只输出类别名称。"),
        ("user", "{text}"),
    ])
    chain = template | llm | StrOutputParser()
    for text in SENTENCES:
        print(f"  '{text[:20]}...' → {chain.invoke({'text': text})}")


def mode2_fewshot():
    """模式2：Few-shot —— 用 LangChain 消息模板"""
    print("\n--- 模式2：Few-shot ---")
    template = ChatPromptTemplate.from_messages([
        ("system", "将文本分类为：正面/负面/中性。只输出类别名称。"),
        ("user", "产品太差了"),
        ("ai", "负面"),
        ("user", "物流很快"),
        ("ai", "正面"),
        ("user", "{text}"),
    ])
    chain = template | llm | StrOutputParser()
    for text in SENTENCES:
        print(f"  '{text[:20]}...' → {chain.invoke({'text': text})}")


def mode3_cot():
    """模式3：CoT —— 用系统提示词注入推理模板"""
    print("\n--- 模式3：CoT ---")
    template = ChatPromptTemplate.from_messages([
        ("system", """你是一个擅长数学推理的助手。
对于每个问题：
1. 列出已知条件
2. 逐步推理
3. 给出最终答案

示例 ——
问题：小明有10元，买了3元的笔和4元的本子，还剩多少？
推理：花费 = 3+4 = 7。剩余 = 10-7 = 3。
答案：3元"""),
        ("user", "{question}"),
    ])
    chain = template | llm | StrOutputParser()
    result = chain.invoke({"question": QUESTION})
    print(f"  问题: {QUESTION}")
    print(f"  回答: {result[:300]}...")


def mode4_react():
    """模式4：ReAct —— 手写循环 + LangChain LLM（上篇已验证）"""
    print("\n--- 模式4：ReAct —— 见 exp3-react/react_loop.py ---")
    print("  LangChain 的 AgentExecutor 底层就是这个 while 循环。")
    print("  上篇已手写实现，此处不重复。")


if __name__ == "__main__":
    mode1_zeroshot()
    mode2_fewshot()
    mode3_cot()
    mode4_react()

"""实验6：LangChain 三个核心概念

ChatModel + PromptTemplate + RunnableSequence（管道 | 运算符）
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ===== 1. ChatModel =====
print("=" * 50)
print("1. ChatModel — 与 LLM 对话的统一接口")
print("=" * 50)

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# 方式 A：直接 .invoke()
response = llm.invoke("用一句话解释什么是 Embedding")
print(f"invoke 返回类型: {type(response).__name__}")
print(f"内容: {response.content[:100]}...\n")

# ===== 2. PromptTemplate =====
print("=" * 50)
print("2. PromptTemplate — 可复用的提示词模板")
print("=" * 50)

# 2a: 简单模板
simple_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，回答风格：{style}。"),
    ("user", "{question}"),
])

messages = simple_template.invoke({
    "role": "幼儿园老师",
    "style": "用小朋友能听懂的语言",
    "question": "为什么天是蓝色的？",
})
print("模板渲染结果:")
print(messages.messages[0].content)
print(messages.messages[1].content[:100], "...\n")

# 2b: Few-shot 模板
fewshot_template = ChatPromptTemplate.from_messages([
    ("system", "将文本分类为：{labels}。"),
    ("user", "{example_1_text}"),
    ("ai", "{example_1_label}"),
    ("user", "{example_2_text}"),
    ("ai", "{example_2_label}"),
    ("user", "{input_text}"),
])

messages = fewshot_template.invoke({
    "labels": "正面/负面/中性",
    "example_1_text": "产品质量太差", "example_1_label": "负面",
    "example_2_text": "物流很快，好评", "example_2_label": "正面",
    "input_text": "性价比很高，推荐购买",
})
print("Few-shot 模板渲染:")
for m in messages.messages:
    print(f"  [{m.type}]: {m.content[:50]}...")
print()

# ===== 3. 管道（| 运算符） =====
print("=" * 50)
print("3. 管道 | — 将组件串联为链")
print("=" * 50)

# LCEL (LangChain Expression Language) 的核心语法
chain = simple_template | llm | StrOutputParser()

result = chain.invoke({
    "role": "脱口秀演员",
    "style": "幽默风趣",
    "question": "为什么程序员喜欢用 dark mode？",
})
print(f"链式调用结果:\n{result}")

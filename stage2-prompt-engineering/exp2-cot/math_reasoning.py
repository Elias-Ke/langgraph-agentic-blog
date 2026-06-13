"""实验2：CoT 思维链对比实验

同一个数学推理题，对比三种方式：
1. 直接问（Zero-shot）
2. 「请逐步思考」（Zero-shot CoT）
3. Few-shot CoT（给带推理步骤的示例）
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ===== 测试题（从简单到困难） =====
questions = [
    # 简单：基本算术推理
    "小明有 15 个苹果，给了小红 3 个，又买了 8 个，现在有多少个？",

    # 中等：需要多步推理
    "一个水池，进水管 3 小时注满，出水管 5 小时放空。两管同时开，几小时注满？",

    # 较难：需要代数思维
    "父亲今年的年龄是儿子的 3 倍，5 年后父亲的年龄是儿子的 2 倍，儿子今年几岁？",

    # 灯泡开关经典题
    "一个房间里有 3 个灯泡，门外有 3 个开关，每个开关控制一个灯泡。"
    "你只能进房间一次，怎么确定哪个开关控制哪个灯泡？",
]

# 正确答案（用于最终验证，不用于评分）
expected_keys = ["20", "7.5", "5", "热"]


def ask_zeroshot(question: str) -> str:
    """Zero-shot：直接问，不要求推理步骤"""
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": question}],
        temperature=0.0,
    )
    return response.choices[0].message.content


def ask_zeroshot_cot(question: str) -> str:
    """Zero-shot CoT：加一句提示，引导模型逐步思考"""
    prompt = f"{question}\n\n请一步一步思考，写出你的推理过程，最后给出答案。"
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


def ask_fewshot_cot(question: str) -> str:
    """Few-shot CoT：给一个带推理步骤的示例，引导模型模仿"""
    system_prompt = """你是一个擅长数学推理的助手。对于每个问题，你会：
1. 列出已知条件
2. 写出解题步骤
3. 给出最终答案

示例——
问题：小明有 10 元，买了 3 元的笔和 4 元的本子，还剩多少元？
推理：
- 已知：总金额 10 元，笔 3 元，本子 4 元
- 步骤 1：总花费 = 3 + 4 = 7 元
- 步骤 2：剩余 = 10 - 7 = 3 元
答案：3 元"""

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def run_experiment(question: str, question_num: int):
    """对一个问题运行三种方式的对比"""
    print(f"\n{'='*60}")
    print(f"问题 {question_num}: {question}")
    print(f"{'='*60}")

    methods = [
        ("Zero-shot（直接问）", ask_zeroshot),
        ("Zero-shot CoT（逐步思考）", ask_zeroshot_cot),
        ("Few-shot CoT（含示例）", ask_fewshot_cot),
    ]

    for method_name, method_fn in methods:
        try:
            start = time.time()
            result = method_fn(question)
            elapsed = time.time() - start

            print(f"\n--- {method_name}（{elapsed:.1f}s）---")
            # 打印完整的推理过程（截断到 350 字符以防刷屏）
            if len(result) > 350:
                print(result[:350] + "\n...(截断，完整输出见日志)")
            else:
                print(result)
        except Exception as e:
            print(f"\n--- {method_name} ---")
            print(f"API 调用失败：{e}")


if __name__ == "__main__":
    for i, question in enumerate(questions):
        run_experiment(question, i + 1)

    print(f"\n{'='*60}")
    print("🔍 观察要点：")
    print("1. Zero-shot CoT 是否比直接问给出了更清晰的推理？")
    print("2. Few-shot CoT 的推理格式是否更规范（模仿了示例）？")
    print("3. 三种方式的答案正确率有何差异？")
    print("4. 「灯泡开关」这种非数学推理题，CoT 还能奏效吗？")

"""实验1：Zero-shot vs Few-shot 分类任务对比

测试 3 个难度递进的分类任务，分别用 0-shot、1-shot、3-shot 测试准确率。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ===== 测试数据 =====

# 任务1（简单）：情感分类——含反讽和混合情感
sentiment_samples = [
    ("产品用了两天就坏了，太差了", "负面"),
    ("物流很快，包装用心，好评", "正面"),
    ("这东西好是好，就是太贵了，不值这个价", "负面"),          # 测试——先扬后抑，表面有正面词
    ("呵呵，客服让我等了半小时就回了一句'好的'", "负面"),     # 测试——反讽
    ("还行吧，毕竟这个价位也不能要求太多", "中性"),           # 测试——勉强中性
    ("以为会很差，结果居然还不错", "正面"),                   # 测试——先抑后扬
]

# 任务2（中等）：意图分类——标签边界模糊，需要领域知识
intent_samples = [
    ("我的订单什么时候能到", "查询物流"),
    ("这个能便宜点吗", "议价"),
    ("我要退货", "售后"),
    ("你们发错颜色了吧，我要的是黑的怎么寄了个白的", "售后"),    # 测试——投诉形式，真实意图是售后
    ("便宜点吧，隔壁店才卖 50", "议价"),                      # 测试——变体表达
    ("已经三天了物流信息还停在转运中心", "查询物流"),          # 测试——投诉口吻，但意图是查物流
]

# 任务3（较难）：逻辑关系——嵌套/隐含逻辑
logic_samples = [
    ("因为下雨，所以比赛取消了", "因果"),
    ("虽然下雨，但是比赛照常进行", "转折"),
    ("如果明天下雨，比赛就取消", "条件"),
    ("虽然知道应该努力，但每次翻开书就犯困，所以到现在还没看完第一章", "因果"),  # 测试——嵌套转折+因果，主逻辑是因果（因为犯困所以没看完）
    ("他这个人，说好吧谈不上，说坏吧也不至于", "转折"),                        # 测试——非标准转折结构
    ("邮件已经发了，但他到现在还没回，也许在忙吧", "因果"),                      # 测试——因果关系藏在推测中
]

# 任务4（最难）：工单优先级——标签是自创的，模型只能从示例学
# 前 4 条是示例池，后 3 条固定测试。每个标签至少出现一次。
ticket_samples = [
    ("线上支付接口全部返回 500，用户无法下单", "T1-紧急"),        # 示例1——阻断业务
    ("后台报表打开需要 30 秒，影响运营日常使用", "T2-重要"),      # 示例2——影响但未阻断
    ("个人中心头像上传后显示模糊，不影响功能", "T3-普通"),        # 示例3——小瑕疵
    ("底部版权年份还是 2024，改不改都行", "T4-低优先级"),         # 示例4——无关紧要
    ("商品搜索接口 P99 延迟从 200ms 涨到了 3 秒", "T2-重要"),    # 测试1——性能下降但未宕机 (T1↔T2 边界)
    ("首页英文 welcome 拼成了 welcom", "T3-普通"),                # 测试2——纯视觉瑕疵
    ("用户登录完全失败，所有用户都登不进去", "T1-紧急"),           # 测试3——阻断核心功能
]


def classify_zeroshot(samples, labels):
    """Zero-shot 分类：不给示例，直接让模型分类"""
    label_str = "/".join(labels)
    prompt = f"将以下文本分类为：{label_str}。只输出类别名称，不要解释。\n\n"
    for text, _ in samples[-3:]:  # 只测最后 3 个
        prompt += f"文本：{text}\n类别："

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def classify_fewshot(samples, labels, n_shots):
    """Few-shot 分类：给 n_shots 个示例"""
    label_str = "/".join(labels)
    system = f"将文本分类为：{label_str}。只输出类别名称。"

    # 构建 few-shot 示例
    examples = "\n".join(
        f"文本：{text}\n类别：{label}"
        for text, label in samples[:n_shots]
    )

    # 测试
    test_texts = "\n".join(
        f"文本：{text}\n类别："
        for text, _ in samples[-3:]
    )

    prompt = f"{system}\n\n示例：\n{examples}\n\n{test_texts}"

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def evaluate(predictions, ground_truth):
    """计算准确率"""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(ground_truth)


def parse_predictions(raw_output):
    """将模型输出解析为预测列表——去掉「类别：」等前缀"""
    preds = []
    for line in raw_output.split("\n"):
        line = line.strip()
        if not line:
            continue
        # 去掉常见的冗余前缀
        for prefix in ["类别：", "类别:", "分类：", "分类:", "预测：", "预测:"]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        preds.append(line.strip())
    return preds


if __name__ == "__main__":
    experiments = [
        ("情感分类（简单）", sentiment_samples, ["正面", "负面", "中性"]),
        ("意图分类——隐含意图（中等）", intent_samples, ["查询物流", "议价", "售后"]),
        ("逻辑关系——无提示词（较难）", logic_samples, ["因果", "转折", "条件"]),
        ("工单优先级——自定义标签（最难）", ticket_samples, ["T1-紧急", "T2-重要", "T3-普通", "T4-低优先级"]),
    ]

    results = {}

    for name, samples, labels in experiments:
        ground_truth = [label for _, label in samples[-3:]]
        print(f"\n{'='*50}")
        print(f"任务：{name}")
        print(f"{'='*50}")

        for mode, n in [("0-shot", 0), ("1-shot", 1), ("3-shot", 3)]:
            try:
                if n == 0:
                    raw = classify_zeroshot(samples, labels)
                else:
                    raw = classify_fewshot(samples, labels, n)

                pred_list = parse_predictions(raw)
                if len(pred_list) == 3:
                    acc = evaluate(pred_list, ground_truth)
                    print(f"  {mode}: 准确率 {acc:.0%}  |  预测: {pred_list}")
                    results[(name, mode)] = acc
                else:
                    print(f"  {mode}: 输出格式异常（期望3行，得到{len(pred_list)}行）: {raw[:100]}...")
            except Exception as e:
                print(f"  {mode}: API 调用失败：{e}")

    # 总结
    print(f"\n{'='*50}")
    print("总结：常识任务 0-shot 够用，自定义标签离不开 Few-shot")
    for name, _, _ in experiments:
        print(f"\n{name}:")
        for mode in ["0-shot", "1-shot", "3-shot"]:
            key = (name, mode)
            if key in results:
                print(f"  {mode}: {results[key]:.0%}")

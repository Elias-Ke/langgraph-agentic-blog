"""ReAct Agent 工具定义"""

import math


def search(query: str) -> str:
    """模拟搜索引擎——用内置知识库代替真实搜索。

    在实际项目中，这里可以替换为真实的搜索 API（如 Bing Search、SerpAPI）
    或向量数据库检索。
    """
    knowledge = [
        # (关键词列表, 答案)
        (["特斯拉", "tesla", "营收", "revenue", "财报"],
         "特斯拉公司（Tesla, Inc.）2024年营收为977亿美元。"),
        (["马斯克", "musk", "elon", "年龄", "age", "几岁", "出生"],
         "埃隆·马斯克（Elon Musk）出生于1971年6月28日，"
         "2026年时年龄为55岁。他是特斯拉和SpaceX的CEO。"),
        (["spacex", "space", "发射", "launch", "星舰", "starship"],
         "SpaceX 2024年完成了134次轨道发射，星舰（Starship）进行了4次试飞。"),
        (["中国gdp", "china gdp", "国内生产总值"],
         "2024年中国GDP为134.9万亿元人民币，同比增长5.0%。"),
    ]

    query_lower = query.lower()
    results = []
    for keywords, value in knowledge:
        if any(kw in query_lower for kw in keywords):
            results.append(value)

    if results:
        return "\n".join(results)
    return f"未找到关于「{query}」的信息。请尝试换一个关键词或更通用的表述。"


def calculator(expression: str) -> str:
    """安全的数学计算器。

    只允许数字、基本运算符和少量数学函数。拒绝包含字母或特殊字符的表达式。
    """
    allowed = set("0123456789+-*/().%^ eEpPiI")
    if not all(c in allowed for c in expression):
        return (
            f"错误：表达式包含不允许的字符。"
            f"只支持数字、基本运算符 (+ - * / ^ %) 和小数点。"
        )

    try:
        # 替换 ^ 为 **
        expr = expression.replace("^", "**")
        # 使用受限的 eval（实际项目建议用 numexpr 或自定义 parser）
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        # 格式化结果：整数直接显示，浮点数保留 2 位
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 4)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

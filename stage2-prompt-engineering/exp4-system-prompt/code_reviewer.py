"""实验4：系统提示词架构化设计

为「代码审查助手」场景设计架构化系统提示词，
对比「随便写的提示词」vs「架构化提示词」的审查质量。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ===== 测试代码（故意留了几个问题） =====
TEST_CODE = '''
def process_orders(orders, discount_rate):
    """处理订单列表，应用折扣"""
    results = []
    for order in orders:
        # 计算折扣后价格
        price = order["price"] * order["quantity"]
        final = price - price * discount_rate

        # 更新订单
        order["total"] = final
        results.append(order)

    # 保存到"数据库"
    with open("/tmp/orders.json", "w") as f:
        f.write(str(results))

    return results


def get_user_email(user_id):
    # 拼接 SQL（危险！）
    query = "SELECT email FROM users WHERE id = " + user_id
    return query
'''

# ===== 方案 A：随便写的提示词 =====
SIMPLE_SYSTEM = "你是一个代码审查助手。请审查以下代码，找出问题。"

# ===== 方案 B：架构化提示词（五层架构） =====
ARCHITECTED_SYSTEM = """# 角色定义
你是一个资深代码审查专家，专注于 Python 后端代码的审查。
你的审查风格是：精确、可操作、不啰嗦。

# 全局约束
- 只审查代码质量和安全性，不评价变量命名风格（除非严重影响可读性）
- 每个问题必须指出：位置（行号）、严重程度、具体修复方案
- 输出使用 Markdown 格式
- 如果代码没有问题，诚实地说「未发现问题」

# 审查维度（按优先级排序）
1. 🔴 安全漏洞：SQL 注入、路径遍历、硬编码密钥、敏感信息泄露
2. 🔴 逻辑错误：边界条件缺失、空值未处理、类型错误
3. 🟡 性能问题：不必要的循环、O(n²) 算法、资源未释放
4. 🟢 代码健壮性：异常处理缺失、输入校验不足、文件操作不安全

# 输出格式
## 审查结果

### 🔴 致命问题（必须修复）
- **位置**: 第 X 行
- **问题**: 具体描述
- **修复方案**:
```python
# 修复后的代码
```

### 🟡 严重问题
（同上格式）

### 🟢 建议优化
（同上格式）

## 总体评分
- 安全性: X/10
- 逻辑正确性: X/10
- 性能: X/10
- 健壮性: X/10"""


def review_code(system_prompt: str, label: str) -> str:
    """用给定的系统提示词审查代码"""
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请审查以下 Python 代码：\n```python\n{TEST_CODE}\n```"},
        ],
        temperature=0.0,
    )
    print(f"\n{'='*60}")
    print(f"审查方式：{label}")
    print(f"{'='*60}")
    print(response.choices[0].message.content)
    return response.choices[0].message.content


if __name__ == "__main__":
    simple_result = review_code(SIMPLE_SYSTEM, "方案 A：随便写的提示词")
    architected_result = review_code(ARCHITECTED_SYSTEM, "方案 B：五层架构化提示词")

    print(f"\n\n{'='*60}")
    print("🔍 对比分析")
    print(f"{'='*60}")
    print("1. 结构化的审查报告 vs 自由格式——哪个更容易执行？")
    print("2. 方案 B 是否发现了方案 A 遗漏的问题？")
    print("3. 方案 B 的修复建议是否更具体（有代码示例）？")
    print("4. 方案 B 的评分体系是否有助于优先级排序？")

    # Token 对比
    print(f"\n📊 Token 消耗对比：")
    print(f"  方案 A 系统提示词: {len(SIMPLE_SYSTEM)} 字符")
    print(f"  方案 B 系统提示词: {len(ARCHITECTED_SYSTEM)} 字符")
    print(f"  方案 B 多用了 {len(ARCHITECTED_SYSTEM) - len(SIMPLE_SYSTEM)} 字符，"
          f"换来了更可预测的输出结构")

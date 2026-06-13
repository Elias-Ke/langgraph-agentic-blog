"""实验5：结构化输出

用 OpenAI response_format 强制 JSON 输出。
对比 JSON Mode（无 Schema）vs JSON Schema strict 模式。
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ===== 测试文本 =====
ARTICLE = """
2024年3月，OpenAI发布了GPT-4o模型，CEO Sam Altman在旧金山的发布会上表示，
新模型在多模态理解和生成速度上有了显著提升。GPT-4o可以同时处理文本、图像和音频输入。

与此同时，Google也发布了Gemini 1.5 Pro，Sundar Pichai在Google I/O大会上展示了
该模型100万Token的超长上下文窗口能力。Anthropic则在6月推出了Claude 3.5 Sonnet，
强调其在代码生成和推理方面的优势。

三家公司都在积极布局AI Agent市场。Microsoft通过Azure AI Studio提供企业级Agent服务，
Google推出了Vertex AI Agent Builder，Anthropic则专注于通过MCP协议标准化Agent工具接入。
"""


def extract_with_json_mode():
    """方式 1：JSON Mode——让模型输出 JSON，但不约束 Schema"""
    system = """从文本中提取实体关系信息。返回严格的 JSON 格式。

输出格式：
{
  "companies": [
    {"name": "公司名", "ceo": "CEO名", "products": ["产品1", "产品2"]}
  ],
  "events": [
    {"date": "时间", "description": "事件描述", "location": "地点"}
  ],
  "relationships": [
    {"from": "实体A", "relation": "关系", "to": "实体B"}
  ]
}"""

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": ARTICLE},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    result = json.loads(response.choices[0].message.content)
    print("✅ JSON Mode 解析成功")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 检查字段完整性
    print("\n📋 字段检查：")
    for field in ["companies", "events", "relationships"]:
        if field in result:
            print(f"  ✅ {field}: {len(result[field])} 条")
        else:
            print(f"  ❌ {field}: 缺失")
    return result


def extract_with_strict_schema():
    """方式 2：JSON Schema strict 模式——严格约束输出结构"""
    system = "从文本中提取实体关系信息。严格按照 Schema 输出。"

    # 定义 strict JSON Schema
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "entity_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "companies": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "ceo": {"type": "string"},
                                "products": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["name", "ceo", "products"],
                            "additionalProperties": False
                        }
                    },
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string"},
                                "description": {"type": "string"},
                                "location": {"type": "string"}
                            },
                            "required": ["date", "description", "location"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["companies", "events"],
                "additionalProperties": False
            }
        }
    }

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": ARTICLE},
        ],
        response_format=schema,
        temperature=0.0,
    )

    result = json.loads(response.choices[0].message.content)
    print("\n✅ JSON Schema strict 模式解析成功")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n📋 Schema 合规检查：")
    for field in ["companies", "events"]:
        if field in result:
            print(f"  ✅ {field}: {len(result[field])} 条（Schema 强制要求）")
        else:
            print(f"  ❌ {field}: 缺失（不应该发生——Schema 要求了 required）")
    if "relationships" in result:
        print(f"  ⚠️  relationships: 存在（但 Schema 未定义——strict 模式应该拒绝）")
    else:
        print(f"  ✅ relationships: 不存在（符合 Schema，strict 模式拦截了）")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("方式 1：JSON Mode（无 Schema 约束）")
    print("=" * 60)
    r1 = extract_with_json_mode()

    print("\n" + "=" * 60)
    print("方式 2：JSON Schema strict 模式")
    print("=" * 60)
    r2 = extract_with_strict_schema()

    print(f"\n{'='*60}")
    print("🔍 对比分析")
    print(f"{'='*60}")
    print("1. JSON Mode 输出了 relationships 字段（按描述），"
          "Schema strict 模式没有（按约束）")
    print("2. Schema strict 保证了字段类型和必需字段——"
          "companies[0].name 一定是 string")
    print("3. JSON Mode 更灵活但不可控，Schema 更严格但需要预先定义结构")
    print("4. Agent 工具调用场景推荐 Schema strict——"
          "工具参数必须精确")

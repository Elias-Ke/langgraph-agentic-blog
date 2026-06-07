"""
手写 Self-Attention — 理解 QKV 的核心计算
用 PyTorch 手动实现单头 Self-Attention，逐行展示每个步骤。

算一遍 softmax(QK^T / sqrt(d_k)) V，
你就不会再觉得 Attention 是「魔法」了。
"""
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,  # (batch, seq_len, d_k)
    K: torch.Tensor,  # (batch, seq_len, d_k)
    V: torch.Tensor,  # (batch, seq_len, d_v)
) -> torch.Tensor:
    """
    核心公式: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    参数:
        Q: Query 矩阵 — "我在找什么？"
        K: Key 矩阵   — "我是什么？"
        V: Value 矩阵 — "我包含什么信息？"

    返回:
        attention_output: 加权后的 Value 矩阵
    """
    d_k = K.shape[-1]

    # Step 1: 计算注意力分数 (Q 和 K 的点积)
    #   结果 shape: (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: 缩放 —— 防止梯度过小
    #   为什么除以 sqrt(d_k)？d_k 越大，点积的方差越大，
    #   softmax 后分布越「尖锐」，梯度越接近 0。
    scores = scores / (d_k ** 0.5)

    # Step 3: softmax —— 把分数变成概率分布
    #   每行的所有值加起来 = 1
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: 加权求和 —— 用注意力权重「混合」Value
    #   注意力高的 Token 贡献更多信息
    attention_output = torch.matmul(attention_weights, V)

    return attention_output, attention_weights


class SelfAttention(torch.nn.Module):
    """带可学习权重矩阵的 Self-Attention（文章 Ch4.6 对应实现）。"""

    def __init__(self, d_model: int, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k
        self.W_q = torch.nn.Linear(d_model, d_k, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_k, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_k, bias=False)

    def forward(
        self, X: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


def main() -> None:
    # 模拟：3 个 Token 的序列，每个 Token 用 4 维向量表示
    torch.manual_seed(42)

    seq_len, d_model = 3, 4
    # 随机初始化 Q, K, V（实际中由输入 X 乘以 W_Q, W_K, W_V 得到）
    Q = torch.randn(1, seq_len, d_model)
    K = torch.randn(1, seq_len, d_model)
    V = torch.randn(1, seq_len, d_model)

    output, weights = scaled_dot_product_attention(Q, K, V)

    print("=" * 50)
    print("手写 Self-Attention 计算结果")
    print("=" * 50)
    print(f"\n输入形状: Q={list(Q.shape)}, K={list(K.shape)}, V={list(V.shape)}")
    print(f"序列长度: {seq_len}, 模型维度: {d_model}")

    print(f"\n注意力权重 (每个 Token 对其它 Token 的关注度):")
    print(weights.squeeze(0))
    print(f"\n每行和 = {weights.sum(dim=-1).squeeze(0)} (应该全是 1.0)")

    print(f"\n注意力输出:")
    print(output.squeeze(0))

    # 检查：输出 = 对 V 的加权平均
    print(f"\n手动计算验证 (加权平均 = 输出？):")
    manual_output = torch.matmul(weights, V)
    print(f"误差: {(output - manual_output).abs().max().item():.10f}")

    # ---- 带可学习权重矩阵的版本（文章 Ch4.6） ----
    print("\n" + "=" * 50)
    print("带可学习权重矩阵的 Self-Attention (文章 Ch4.6)")
    print("=" * 50)
    d_model, d_k = 8, 4
    sa = SelfAttention(d_model, d_k)
    X = torch.randn(1, 3, d_model)  # batch=1, seq_len=3, d_model=8
    out, attn = sa(X)
    print(f"输入形状: {list(X.shape)}")
    print(f"输出形状: {list(out.shape)}")
    print(f"注意力权重:\n{attn.squeeze(0)}")


if __name__ == "__main__":
    main()

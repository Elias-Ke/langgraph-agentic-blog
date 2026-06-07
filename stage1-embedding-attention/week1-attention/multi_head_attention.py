"""
多头注意力 (NumPy 实现) — 对应文章 Ch5.1
多个「头」从不同视角同时评估同一段文字。
"""
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)
    return np.matmul(attn, V), attn


class MultiHeadAttention:
    """多头注意力"""

    def __init__(self, d_model: int, num_heads: int) -> None:
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除!"
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """(seq_len, d_model) → (num_heads, seq_len, d_k)"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return np.transpose(x, (1, 0, 2))

    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """(num_heads, seq_len, d_k) → (seq_len, d_model)"""
        x = np.transpose(x, (1, 0, 2))
        return x.reshape(x.shape[0], self.d_model)

    def forward(self, X: np.ndarray) -> np.ndarray:
        seq_len = X.shape[0]
        Q = np.matmul(X, self.W_q)
        K = np.matmul(X, self.W_k)
        V = np.matmul(X, self.W_v)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        head_outputs = []
        for i in range(self.num_heads):
            head_out, _ = scaled_dot_product_attention(Q[i], K[i], V[i])
            head_outputs.append(head_out)

        heads = np.stack(head_outputs, axis=0)
        output = self.combine_heads(heads)
        return np.matmul(output, self.W_o)


def main() -> None:
    print("=" * 55)
    print("Ch5.1: 多头注意力测试")
    print("=" * 55)
    d_model, num_heads = 8, 2
    mha = MultiHeadAttention(d_model, num_heads)
    X = np.random.randn(5, d_model)

    output = mha.forward(X)
    print(f"输入形状:   {X.shape}  (5 个 Token, 每个 {d_model} 维)")
    print(f"多头数:     {num_heads}, 每个头维度: {mha.d_k}")
    print(f"输出形状:   {output.shape}  (保持 d_model 不变)")


if __name__ == "__main__":
    main()

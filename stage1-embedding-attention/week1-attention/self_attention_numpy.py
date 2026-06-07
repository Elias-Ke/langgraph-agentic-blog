"""
手写 Self-Attention (NumPy 版) — 对应文章 Ch4.5
不依赖 PyTorch，只用 NumPy 逐行展示计算过程。
"""
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 Softmax"""
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    核心公式: Attention(Q,K,V) = softmax(Q·K^T / sqrt(d_k)) · V

    Q: Query  (seq_len, d_k)
    K: Key    (seq_len, d_k)
    V: Value  (seq_len, d_k)
    """
    d_k = Q.shape[-1]

    # Step 1: Q·K^T
    scores = np.matmul(Q, K.T)
    print(f"Q·K^T 分数矩阵:\n{scores}")

    # Step 2: 缩放
    scores = scores / np.sqrt(d_k)
    print(f"\n缩放后 (除以 sqrt({d_k}) = {np.sqrt(d_k):.4f}):\n{scores}")

    # Step 3: Softmax
    attention_weights = softmax(scores, axis=-1)
    print(f"\n注意力权重 (Softmax 后):\n{attention_weights}")
    print(f"每行的和 (应为 1): {np.sum(attention_weights, axis=-1)}")

    # Step 4: 加权求和
    output = np.matmul(attention_weights, V)
    return output, attention_weights


def main() -> None:
    np.random.seed(42)
    seq_len, d_k = 3, 4

    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)

    print("=" * 55)
    print("Ch4.5: 手写 Self-Attention (NumPy 版)")
    print("=" * 55)
    print(f"\nQ:\n{Q}")
    print(f"\nK:\n{K}")
    print(f"\nV:\n{V}")
    print()

    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"\n最终输出 (每个 Token 的新表示):\n{output}")


if __name__ == "__main__":
    main()

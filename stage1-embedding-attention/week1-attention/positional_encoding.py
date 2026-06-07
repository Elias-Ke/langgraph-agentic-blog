"""
正弦/余弦位置编码 — 对应文章 Ch5.2
不同频率的正弦/余弦波 = "连续版二进制"，给 Transformer 注入位置信息。
"""
import numpy as np


def get_positional_encoding(pos: int, d_model: int) -> np.ndarray:
    """计算某个位置的位置编码"""
    PE = np.zeros(d_model)
    for i in range(d_model // 2):
        angle = pos / (10000 ** (2 * i / d_model))
        PE[2 * i] = np.sin(angle)
        PE[2 * i + 1] = np.cos(angle)
    return PE


def main() -> None:
    print("=" * 55)
    print("Ch5.2: 正弦/余弦位置编码")
    print("=" * 55)

    d_model, max_len = 128, 50
    dimensions = [0, 2, 4, 8, 16, 32]

    print(f"模型维度: {d_model}")
    print(f"展示位置 0~{max_len-1} 的编码值\n")

    for dim in dimensions:
        wave = [get_positional_encoding(pos, d_model)[dim] for pos in range(max_len)]
        # 用文本画一个简化的波形
        chars = []
        for v in wave:
            if v > 0.5: chars.append("█")
            elif v > 0: chars.append("▄")
            elif v > -0.5: chars.append("▁")
            else: chars.append("▔")
        print(f"dim {dim:>3} (2i={dim}): {''.join(chars[:50])}")

    print(f"\n观察：dim 0 振荡最快 (高频 → 细粒度位置), dim 32 最慢 (低频 → 大尺度位置)")
    print("这就是「连续版二进制」——每个维度以不同频率交替。")

    # 也可用 matplotlib 画波形（可选）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 6))
        for dim in dimensions:
            wave = [get_positional_encoding(pos, d_model)[dim] for pos in range(max_len)]
            plt.plot(range(max_len), wave, label=f"dim {dim} (2i={dim})")
        plt.title("Positional Encoding: 不同维度的波形 (频率递减)")
        plt.xlabel("位置 pos")
        plt.ylabel("编码值")
        plt.legend()
        plt.grid(True)
        path = "/tmp/positional_encoding.png"
        plt.savefig(path, dpi=150)
        print(f"\n波形图已保存至: {path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()

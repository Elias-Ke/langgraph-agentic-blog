"""
Embedding 本质演示 — 对应文章 Ch1.4 全部三个实验
实验1: 基础句子相似度 (BGE 中文模型)
实验2: 向量运算语义类比 (国王 - 男人 + 女人 ≈ 女王)
实验3: t-SNE 降维可视化
"""
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    print("加载 BGE 中文模型... (首次运行需下载，约 100MB)")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    # ---- 实验 1：基础句子相似度 ----
    print("=" * 55)
    print("实验 1：基础句子相似度")
    print("=" * 55)
    sentences = [
        "猫是一种常见的宠物",
        "小猫很可爱，很多人养在家里",
        "狗也是人类的好朋友",
        "今天天气真好啊",
        "深度神经网络使用反向传播来训练",
    ]
    embeddings = model.encode(sentences)
    anchor = embeddings[0]
    print(f"锚点: 「{sentences[0]}」")
    print(f"向量维度: {embeddings.shape[1]}")
    for sent, emb in zip(sentences, embeddings):
        sim = cosine_similarity(anchor, emb)
        marker = "  ← 锚点" if sent == sentences[0] else ""
        print(f"  {sim:>7.4f}  {sent}{marker}")

    # ---- 实验 2：语义类比 (国王 - 男人 + 女人 ≈ 女王) ----
    print(f"\n{'=' * 55}")
    print("实验 2：向量运算语义类比")
    print("=" * 55)
    king   = model.encode("国王", normalize_embeddings=True)
    man    = model.encode("男人", normalize_embeddings=True)
    woman  = model.encode("女人", normalize_embeddings=True)
    queen  = model.encode("女王", normalize_embeddings=True)

    result = king - man + woman
    result = result / np.linalg.norm(result)
    queen_norm = queen / np.linalg.norm(queen)
    sim = float(np.dot(result, queen_norm))
    print(f"国王 - 男人 + 女人 与 女王 的余弦相似度: {sim:.4f}")
    print("(越接近 1.0 说明向量运算捕捉到了语义关系)")

    # ---- 实验 3：t-SNE 降维可视化 ----
    print(f"\n{'=' * 55}")
    print("实验 3：t-SNE 降维可视化")
    print("=" * 55)
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        words = ["国王", "女王", "男人", "女人", "男孩", "女孩",
                 "苹果", "香蕉", "橘子", "汽车", "公交车", "火车"]
        emb = model.encode(words)
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        emb_2d = tsne.fit_transform(emb)

        plt.figure(figsize=(10, 8))
        for i, word in enumerate(words):
            x, y = emb_2d[i]
            plt.scatter(x, y)
            plt.annotate(word, (x, y), fontsize=12)
        plt.title("Word Embeddings visualized with t-SNE (BGE Chinese)")
        output_path = "/tmp/embedding_tsne.png"
        plt.savefig(output_path, dpi=150)
        print(f"图片已保存至: {output_path}")
        print("（如果是在命令行运行，图片已保存；在 IDE 中运行请用 plt.show()）")
    except ImportError:
        print("跳过: 需要 scikit-learn 和 matplotlib (uv sync 已包含)")


if __name__ == "__main__":
    main()

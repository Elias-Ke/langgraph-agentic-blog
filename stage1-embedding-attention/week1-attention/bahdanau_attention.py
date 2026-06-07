"""
Bahdanau (加法) Attention — 对应文章 Ch3.5
用于 Seq2Seq 解码器，通过 score = v^T·tanh(W1·h_dec + W2·h_enc) 计算注意力。
"""
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """Bahdanau (加法) Attention"""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V  = nn.Linear(hidden_size, 1)

    def forward(
        self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        decoder_hidden:  (batch, hidden)   — 解码器上一步隐藏状态
        encoder_outputs: (batch, seq_len, hidden) — 编码器所有隐藏状态
        返回: context (batch, hidden), attention_weights (batch, seq_len)
        """
        dec_hidden = decoder_hidden.unsqueeze(1)  # (batch, 1, hidden)

        # 打分: v^T · tanh(W1·h_dec + W2·h_enc)
        scores = self.V(torch.tanh(self.W1(dec_hidden) + self.W2(encoder_outputs)))
        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Softmax 归一化
        attention_weights = torch.softmax(scores, dim=-1)

        # 加权求和: Σ α_s · h_s
        context = torch.sum(
            attention_weights.unsqueeze(-1) * encoder_outputs, dim=1
        )  # (batch, hidden)

        return context, attention_weights


def main() -> None:
    print("=" * 55)
    print("Ch3.5: Bahdanau Attention 测试")
    print("=" * 55)
    attn = BahdanauAttention(hidden_size=16)

    decoder_hidden  = torch.randn(4, 16)     # batch=4
    encoder_outputs = torch.randn(4, 10, 16) # batch=4, seq_len=10

    context, weights = attn(decoder_hidden, encoder_outputs)

    print(f"上下文向量形状:   {list(context.shape)}  (batch=4, hidden=16)")
    print(f"注意力权重形状:   {list(weights.shape)}   (batch=4, seq_len=10)")
    print(f"\n某个 batch 的注意力权重 (应为和为 1):")
    print(weights[0])
    print(f"权重和: {weights[0].sum().item():.4f}")


if __name__ == "__main__":
    main()

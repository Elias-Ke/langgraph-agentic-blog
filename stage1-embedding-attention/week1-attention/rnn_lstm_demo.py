"""
RNN / LSTM / GRU 基础演示 — 对应文章 Ch3.1~3.3
- 最简 RNN 前向传播
- LSTM vs RNN 长距离依赖对比
- GRU 一句话切换
"""
import torch
import torch.nn as nn


def demo_rnn() -> None:
    """Ch3.1: 最简 RNN"""
    print("=" * 50)
    print("Ch3.1: 最简 RNN 前向传播")
    print("=" * 50)
    rnn = nn.RNN(input_size=8, hidden_size=16, num_layers=1, batch_first=True)
    X = torch.randn(2, 5, 8)  # batch=2, seq_len=5, input=8
    output, h_n = rnn(X)
    print(f"输入形状:    {list(X.shape)}")
    print(f"output 形状: {list(output.shape)}  (每个时间步都有输出)")
    print(f"h_n 形状:    {list(h_n.shape)}     (最后一步的隐藏状态)")
    print("output vs h_n: output 含每步输出，h_n 只有最后一步")


def demo_lstm_vs_rnn() -> None:
    """Ch3.3: LSTM vs RNN 长距离依赖 (简化版)"""
    print(f"\n{'=' * 50}")
    print("Ch3.3: LSTM vs RNN 长距离依赖对比 (简化演示)")
    print("=" * 50)

    # RNN
    rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
    # LSTM
    lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)

    X = torch.randn(4, 20, 1)  # batch=4, seq_len=20
    rnn_out, _ = rnn(X)
    lstm_out, _ = lstm(X)

    print(f"输入序列长度: 20")
    print(f"RNN  输出形状:  {list(rnn_out.shape)}")
    print(f"LSTM 输出形状:  {list(lstm_out.shape)}")
    print()
    print("关键差异（概念层面，本 demo 只展示结构）：")
    print("- RNN:  每个时间步 hidden state 完全覆盖 → 旧信息迅速丢失")
    print("- LSTM: Cell State 加法更新 → 梯度沿「高速公路」传导")
    print("- LSTM 能捕捉 ~20 步依赖，RNN 约 10 步后就「忘记」开头")


def demo_gru() -> None:
    """Ch3.3 附: GRU — 一句话替换"""
    print(f"\n{'=' * 50}")
    print("Ch3.3: GRU — PyTorch 一句话切换")
    print("=" * 50)
    gru = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
    X = torch.randn(2, 5, 8)
    output, h_n = gru(X)
    print(f"GRU 输出形状: {list(output.shape)}")
    print("把 nn.RNN 或 nn.LSTM 换成 nn.GRU，代码其余不变")


def main() -> None:
    demo_rnn()
    demo_lstm_vs_rnn()
    demo_gru()


if __name__ == "__main__":
    main()

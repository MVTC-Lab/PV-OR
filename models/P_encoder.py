import torch
import torch.nn as nn

class PromptEncoder(nn.Module):
    def __init__(self, prompt_len, embed_dim, hidden_dim=768, num_P = 8):
        super().__init__()
        self.prompt_len = prompt_len
        # 伪 token 的 embedding (可训)
        self.pseudo_emb = nn.Embedding(prompt_len, embed_dim)
        # MLP + LSTM 编码器
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=1, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, batch_size):
        # 获取 prompt 初始 embedding
        x = self.pseudo_emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.mlp(x)
        x, _ = self.lstm(x)
        x = self.ln(x)
        return x  # shape = (B, M, D)

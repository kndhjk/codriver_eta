# models/wdr.py
import torch
import torch.nn as nn

class SimpleWDR(nn.Module):
    """
    简化 WDR：wide(交叉特征) + deep(embeddings + MLP)
    输入：link_id_idx, time_idx  (均为编码后的 int64 索引)
    目标：avg_speed_mps (回归, m/s)
    """
    def __init__(self,
                 n_links: int,
                 n_times: int,
                 emb_dim_link: int = 32,
                 emb_dim_time: int = 8,
                 mlp_hidden=(128, 64),
                 wide_cross: bool = True):
        super().__init__()
        self.link_emb = nn.Embedding(n_links, emb_dim_link)
        self.time_emb = nn.Embedding(n_times, emb_dim_time)

        self.wide_cross = wide_cross
        if wide_cross:
            # 简单的 wide：对 (link, time) 做 one-hot 交叉的等价：用一个小表学习交叉偏置
            self.cross_bias = nn.Embedding(n_links * n_times, 1)
        in_dim = emb_dim_link + emb_dim_time
        layers = []
        last = in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, link_idx, time_idx):
        # link_idx, time_idx: (B,) int64
        le = self.link_emb(link_idx)
        te = self.time_emb(time_idx)
        x = torch.cat([le, te], dim=-1)
        y = self.mlp(x).squeeze(-1)  # (B,)

        if self.wide_cross:
            # 交叉索引：link * n_times + time
            n_times = self.time_emb.num_embeddings
            cross_idx = link_idx * n_times + time_idx
            y = y + self.cross_bias(cross_idx).squeeze(-1)
        return y

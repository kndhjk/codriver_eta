# models/codriver.py
import torch
import torch.nn as nn

class CoDriverSimple(nn.Module):
    """
    link_emb + time_emb + driver_emb -> MLP 回归 (avg_speed_mps)
    """
    def __init__(self,
                 n_links:int, n_times:int, n_drivers:int,
                 emb_dim_link:int=32, emb_dim_time:int=8, emb_dim_driver:int=16,
                 mlp_hidden=(128,64),
                 wide_cross:bool=True):
        super().__init__()
        self.link_emb   = nn.Embedding(n_links, emb_dim_link)
        self.time_emb   = nn.Embedding(n_times, emb_dim_time)
        self.driver_emb = nn.Embedding(n_drivers, emb_dim_driver)

        self.wide_cross = wide_cross
        if wide_cross:
            self.cross_bias = nn.Embedding(n_links * n_times, 1)

        in_dim = emb_dim_link + emb_dim_time + emb_dim_driver
        layers = []
        last = in_dim
        for h in mlp_hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, link_idx, time_idx, driver_idx):
        le = self.link_emb(link_idx)
        te = self.time_emb(time_idx)
        de = self.driver_emb(driver_idx)
        x  = torch.cat([le, te, de], dim=-1)
        y  = self.mlp(x).squeeze(-1)

        if self.wide_cross:
            n_times = self.time_emb.num_embeddings
            cross_idx = link_idx * n_times + time_idx
            y = y + self.cross_bias(cross_idx).squeeze(-1)
        return y

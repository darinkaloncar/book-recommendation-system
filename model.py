import torch
import torch.nn as nn

class HybridNCF(nn.Module):
    """
    Hybrid Neural Collaborative Filtering
    Embeddings for user, item, author, publisher + numeric features (user age, book year_norm).
    Output: scalar rating (regression). Use MSELoss for training and take sqrt for RMSE reporting.
    """
    def __init__(self,
                 n_users,
                 n_books,
                 n_authors,
                 n_publishers,
                 user_emb_dim=32,
                 item_emb_dim=32,
                 author_emb_dim=16,
                 pub_emb_dim=16,
                 mlp_layers=(128, 64, 32),
                 numeric_dim=2,
                 dropout=0.2):
        super().__init__()
        # embeddings
        self.user_emb = nn.Embedding(max(1, n_users), user_emb_dim)
        self.item_emb = nn.Embedding(max(1, n_books), item_emb_dim)
        self.author_emb = nn.Embedding(max(1, n_authors), author_emb_dim)
        self.pub_emb = nn.Embedding(max(1, n_publishers), pub_emb_dim)

        concat_dim = user_emb_dim + item_emb_dim + author_emb_dim + pub_emb_dim + numeric_dim

        layers = []
        in_dim = concat_dim
        for out_dim in mlp_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))  # final scalar output

        self.mlp = nn.Sequential(*layers)

        # initialize embeddings
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.author_emb.weight, std=0.01)
        nn.init.normal_(self.pub_emb.weight, std=0.01)

    def forward(self, user_idx, item_idx, author_idx, pub_idx, user_age, book_year_norm):
        """
        Inputs:
          - user_idx, item_idx, author_idx, pub_idx: LongTensor (batch,)
          - user_age, book_year_norm: FloatTensor (batch,)
        Returns:
          - out: FloatTensor (batch,) predicted rating
        """
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        a = self.author_emb(author_idx)
        p = self.pub_emb(pub_idx)

        numerics = torch.stack([user_age, book_year_norm], dim=1)  # (batch,2)

        x = torch.cat([u, i, a, p, numerics], dim=1)
        out = self.mlp(x).squeeze(1)
        return out


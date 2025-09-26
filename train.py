import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import argparse
import numpy as np
import pandas as pd

from data_prep import preprocess, load_book_crossing
from model import HybridNCF

class RecDataset(Dataset):
    """
    PyTorch Dataset for Book-Crossing / synthetic data.
    Safely handles missing users or books.
    """
    def __init__(self, ratings_df, users_df, books_df, author_map, pub_map):
        self.ratings = ratings_df.reset_index(drop=True)
        # index by mapped indices for safe .loc access
        self.users = users_df.set_index('user_idx', drop=False)
        self.books = books_df.set_index('book_idx', drop=False)
        self.author_map = author_map
        self.pub_map = pub_map

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        row = self.ratings.iloc[idx]
        u_idx = int(row['user_idx'])
        b_idx = int(row['book_idx'])
        rating = float(row['rating'])

        # safe access for book
        if b_idx in self.books.index:
            book_row = self.books.loc[b_idx]
        else:
            book_row = pd.Series({'author': 'Unknown', 'publisher': 'Unknown', 'year_norm': 0.0})

        author = book_row.get('author', 'Unknown')
        pub = book_row.get('publisher', 'Unknown')
        author_idx = self.author_map.get(author, 0)
        pub_idx = self.pub_map.get(pub, 0)

        # safe access for user
        if u_idx in self.users.index:
            user_row = self.users.loc[u_idx]
        else:
            user_row = pd.Series({'age': -1})

        age = float(user_row.get('age', -1))
        year_norm = float(book_row.get('year_norm', 0.0))

        return {
            'user_idx': u_idx,
            'book_idx': b_idx,
            'author_idx': author_idx,
            'pub_idx': pub_idx,
            'age': age,
            'year_norm': year_norm,
            'rating': rating
        }


def collate_fn(batch):
    import torch
    return {
        'user_idx': torch.tensor([b['user_idx'] for b in batch], dtype=torch.long),
        'book_idx': torch.tensor([b['book_idx'] for b in batch], dtype=torch.long),
        'author_idx': torch.tensor([b['author_idx'] for b in batch], dtype=torch.long),
        'pub_idx': torch.tensor([b['pub_idx'] for b in batch], dtype=torch.long),
        'age': torch.tensor([b['age'] for b in batch], dtype=torch.float),
        'year_norm': torch.tensor([b['year_norm'] for b in batch], dtype=torch.float),
        'rating': torch.tensor([b['rating'] for b in batch], dtype=torch.float),
    }

def build_maps(books_df):
    authors = list(books_df['author'].unique())
    pubs = list(books_df['publisher'].unique())
    author_map = {a: i for i, a in enumerate(authors)}
    pub_map = {p: i for i, p in enumerate(pubs)}
    return author_map, pub_map

def train_loop(args):
    # Load data
    users_raw, books_raw, ratings_raw = load_book_crossing(args.data_dir)
    users, books, ratings, user_map, book_map = preprocess(users_raw, books_raw, ratings_raw,
                                                           min_user_ratings=args.min_user_ratings,
                                                           min_book_ratings=args.min_book_ratings)

    # build author/publisher maps from books table
    author_map, pub_map = build_maps(books)

    # train/test split
    train_df, test_df = train_test_split(ratings, test_size=args.test_size, random_state=42)
    train_ds = RecDataset(train_df, users, books, author_map, pub_map)
    test_ds = RecDataset(test_df, users, books, author_map, pub_map)

    # Wraps datasets in batches.
    # collate_fn converts list-of-samples → batch-of-tensors.
    # shuffle=True for training ensures randomized batches.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HybridNCF(
        n_users=len(user_map),
        n_books=len(book_map),
        n_authors=len(author_map),
        n_publishers=len(pub_map),
        user_emb_dim=args.user_emb_dim,
        item_emb_dim=args.item_emb_dim,
        author_emb_dim=args.author_emb_dim,
        pub_emb_dim=args.pub_emb_dim,
        mlp_layers=tuple(int(x) for x in args.mlp_layers.split(',')) if isinstance(args.mlp_layers, str) else tuple(args.mlp_layers)
    ).to(device) #moves model to GPU/CPU

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss() #MSELoss: mean squared error for rating prediction (regression).

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            preds = model(batch['user_idx'].to(device),
                          batch['book_idx'].to(device),
                          batch['author_idx'].to(device),
                          batch['pub_idx'].to(device),
                          batch['age'].to(device),
                          batch['year_norm'].to(device))
            loss = loss_fn(preds, batch['rating'].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch['rating'].shape[0]
            n_samples += batch['rating'].shape[0]

        train_rmse = (total_loss / max(1, n_samples)) ** 0.5
        print(f"Epoch {epoch} Train RMSE: {train_rmse:.4f}")

        # basic test evaluation each epoch
        model.eval()
        with torch.no_grad():
            sq = 0.0
            n = 0
            for batch in test_loader:
                preds = model(batch['user_idx'].to(device),
                              batch['book_idx'].to(device),
                              batch['author_idx'].to(device),
                              batch['pub_idx'].to(device),
                              batch['age'].to(device),
                              batch['year_norm'].to(device))
                diff = (preds.cpu().numpy() - batch['rating'].numpy())
                sq += (diff ** 2).sum()
                n += len(diff)
            test_rmse = (sq / max(1, n)) ** 0.5
        print(f"Epoch {epoch} Test RMSE: {test_rmse:.4f}")

        # save checkpoint
        ckpt_path = f"checkpoint_epoch{epoch}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, ckpt_path)

    # final save
    torch.save(model.state_dict(), "models/model_final.pth")
    print("Training finished. Model saved to model_final.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/book_crossing')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--user_emb_dim', type=int, default=32)
    parser.add_argument('--item_emb_dim', type=int, default=32)
    parser.add_argument('--author_emb_dim', type=int, default=16)
    parser.add_argument('--pub_emb_dim', type=int, default=16)
    parser.add_argument('--mlp_layers', type=str, default='128,64,32', help='Comma-separated sizes for MLP')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--min_user_ratings', type=int, default=5)
    parser.add_argument('--min_book_ratings', type=int, default=5)
    args = parser.parse_args()
    train_loop(args)


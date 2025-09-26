import torch
from torch.utils.data import DataLoader
from train import RecDataset, collate_fn, build_maps
from data_prep import load_book_crossing, preprocess
from model import HybridNCF
from evaluate import rmse, precision_recall_at_k
import numpy as np

# ----------------------------
# Load data
# ----------------------------
users, books, ratings = load_book_crossing("data")
users, books, ratings, user_map, book_map = preprocess(users, books, ratings)

# Build author/publisher maps
author_map, pub_map = build_maps(books)

# Create test dataset & loader
test_dataset = RecDataset(ratings, users, books, author_map, pub_map)
test_loader = DataLoader(test_dataset, batch_size=256, collate_fn=collate_fn)

# ----------------------------
# Load model
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridNCF(
    n_users=len(user_map),
    n_books=len(book_map),
    n_authors=len(author_map),
    n_publishers=len(pub_map)
).to(device)
model.load_state_dict(torch.load("models/model_final.pth"))
model.eval()

# ----------------------------
# Evaluate RMSE
# ----------------------------
all_preds = []
all_ratings = []

with torch.no_grad():
    for batch in test_loader:
        preds = model(batch['user_idx'].to(device),
                      batch['book_idx'].to(device),
                      batch['author_idx'].to(device),
                      batch['pub_idx'].to(device),
                      batch['age'].to(device),
                      batch['year_norm'].to(device))
        all_preds.extend(preds.cpu().numpy())
        all_ratings.extend(batch['rating'].numpy())

print("Test RMSE:", rmse(np.array(all_ratings), np.array(all_preds)))

# ----------------------------
# Example: precision/recall at k
# ----------------------------
# Convert predictions to format for precision_recall_at_k
# Here we consider ratings >= 7 as relevant
user_preds = {}
for batch in test_loader:
    with torch.no_grad():
        scores = model(batch['user_idx'].to(device),
                       batch['book_idx'].to(device),
                       batch['author_idx'].to(device),
                       batch['pub_idx'].to(device),
                       batch['age'].to(device),
                       batch['year_norm'].to(device)).cpu().numpy()
    for u, b, s, r in zip(batch['user_idx'].numpy(), batch['book_idx'].numpy(), scores, batch['rating'].numpy()):
        user_preds.setdefault(u, []).append((b, s, r >= 7))

prec, rec = precision_recall_at_k(user_preds, k=10)
print(f"Precision@10: {prec:.4f}, Recall@10: {rec:.4f}")
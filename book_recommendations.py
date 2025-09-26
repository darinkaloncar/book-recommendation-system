import torch
from data_prep import load_book_crossing, preprocess
from model import HybridNCF  # your trained model class

# 1. Load and preprocess data
data_path = 'data'
users, books, ratings = load_book_crossing(data_path)
users_sub, books_sub, ratings_sub, user_map, book_map = preprocess(users, books, ratings)

# 2. Build author/publisher maps for embeddings
author_map = {a: i for i, a in enumerate(books_sub['author'].unique())}
books_sub['author_idx'] = books_sub['author'].map(author_map)
pub_map = {p: i for i, p in enumerate(books_sub['publisher'].unique())}
books_sub['pub_idx'] = books_sub['publisher'].map(pub_map)
print("ovde")
print(len(books_sub))

# 3. Load your trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridNCF(
    n_users=len(user_map),
    n_books=len(book_map),
    n_authors=len(author_map),
    n_publishers=len(pub_map)
)
model.load_state_dict(torch.load('models/model_final.pth', map_location=device))
model.to(device)
model.eval()

# 4. Make recommendations for a user
user_id_to_test = list(user_map.keys())[0]  # replace with real user ID
if user_id_to_test not in user_map:
    print(f"User {user_id_to_test} not found in dataset")
    exit()

print("ovde1")
print(len(books_sub))

u_idx = user_map[user_id_to_test]

# compute scores for all books
book_indices = torch.tensor(books_sub['book_idx'].values, device=device)
user_indices = torch.tensor([u_idx] * len(book_indices), device=device)
author_indices = torch.tensor(books_sub['author_idx'].values, device=device)
pub_indices = torch.tensor(books_sub['pub_idx'].values, device=device)
user_age = torch.tensor([users_sub.loc[users_sub['user_idx'] == u_idx, 'age'].values[0]] * len(book_indices), device=device, dtype=torch.float32)
book_year_norm = torch.tensor(books_sub['year_norm'].values, device=device, dtype=torch.float32)

with torch.no_grad():
    scores = model(user_indices, book_indices, author_indices, pub_indices, user_age, book_year_norm)


print("ovde2")
print(len(books_sub))
# get top-N recommendations
topN = 11
top_indices = torch.topk(scores, min(topN, len(books_sub))).indices.cpu().numpy()

print(f"Top {topN} book recommendations for user {user_id_to_test}:")
top_books_sub = books_sub.iloc[top_indices]  # directly get rows
for _, book_info in top_books_sub.iterrows():
    print(f"{book_info['book_id']} - {book_info['author']} - {book_info['publisher']} - {book_info['year']}")
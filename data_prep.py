import os
import pandas as pd
import numpy as np

def load_book_crossing(path):
    """
    Expects files named:
      - books.csv
      - ratings.csv
      - users.csv
    placed inside `path`.
    """
    books_fp = os.path.join(path, 'books.csv')
    ratings_fp = os.path.join(path, 'ratings.csv')
    users_fp = os.path.join(path, 'users.csv')
    if not (os.path.exists(books_fp) and os.path.exists(ratings_fp) and os.path.exists(users_fp)):
        raise FileNotFoundError('Book-Crossing files not found in %s' % path)

    # read CSVs safely
    books = pd.read_csv(books_fp, sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    ratings = pd.read_csv(ratings_fp, sep=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv(users_fp, sep=';', encoding='latin-1', on_bad_lines='skip')

    # normalize column names
    books.columns = [c.strip() for c in books.columns]
    ratings.columns = [c.strip() for c in ratings.columns]
    users.columns = [c.strip() for c in users.columns]

    books.rename(columns={
        'ISBN': 'book_id',
        'Book-Author': 'author',
        'Year-Of-Publication': 'year',
        'Publisher': 'publisher'
    }, inplace=True)

    ratings.rename(columns={
        'User-ID': 'user_id',
        'ISBN': 'book_id',
        'Book-Rating': 'rating'
    }, inplace=True)

    users.rename(columns={
        'User-ID': 'user_id',
        'Age': 'age'
    }, inplace=True)

    # safely convert numeric columns
    if 'year' in books.columns:
        books['year'] = pd.to_numeric(books['year'], errors='coerce').fillna(0).astype(float)
    if 'age' in users.columns:
        users['age'] = pd.to_numeric(users['age'], errors='coerce').fillna(-1).astype(float)

    return users, books, ratings

def preprocess(users, books, ratings, min_user_ratings=5, min_book_ratings=5):
    """
    for df in (users, books, ratings):
        df.columns = [c.strip() for c in df.columns]
    """

    # drop zero ratings
    if 'rating' in ratings.columns:
        ratings = ratings[ratings['rating'] != 0].copy()
    else:
        raise ValueError("ratings DataFrame missing 'rating' column")

    # filter users/books by count
    # remove all users who have less than min_user_ratings rated books
    # remove all books which have less than min_book_ratings ratings
    user_counts = ratings.groupby('user_id').size()
    book_counts = ratings.groupby('book_id').size()
    valid_users = set(user_counts[user_counts >= min_user_ratings].index)
    valid_books = set(book_counts[book_counts >= min_book_ratings].index)
    print(f"Number of valid users:", valid_users, user_counts)

    # if all books and users are removed, we undo the previous step
    if len(valid_users) == 0 or len(valid_books) == 0:
        valid_users = set(ratings['user_id'].unique())
        valid_books = set(ratings['book_id'].unique())

    ratings = ratings[ratings['user_id'].isin(valid_users) & ratings['book_id'].isin(valid_books)].copy()

    # remap to contiguous indices
    # due to book and user removal indices are bound to be messed up eg. unique_users = [10, 23, 42]
    unique_users = ratings['user_id'].unique()
    unique_books = ratings['book_id'].unique()
    user_map = {old: new for new, old in enumerate(unique_users)}
    book_map = {old: new for new, old in enumerate(unique_books)}
    print(f"Number of unique users:",unique_users, user_map)

    # add new column which has good indices, user_map = {10:0, 23:1, 42:2}
    ratings['user_idx'] = ratings['user_id'].map(user_map)
    ratings['book_idx'] = ratings['book_id'].map(book_map)

    # after adding new indices to ratings dataframe, we make new dataframes for books and users,
    # that only contain valid info
    books_sub = books[books['book_id'].isin(book_map.keys())].copy()
    books_sub['book_idx'] = books_sub['book_id'].map(book_map)

    users_sub = users[users['user_id'].isin(user_map.keys())].copy()
    users_sub['user_idx'] = users_sub['user_id'].map(user_map)

    # age handling
    users_sub['age'] = pd.to_numeric(users_sub.get('age', -1), errors='coerce').fillna(-1).astype(float)

    # year normalization
    books_sub['year'] = pd.to_numeric(books_sub.get('year', 0), errors='coerce').fillna(0)
    mean = books_sub['year'].mean() if len(books_sub) > 0 else 0.0
    std = books_sub['year'].std() if len(books_sub) > 0 else 1.0
    books_sub['year_norm'] = (books_sub['year'] - mean) / (std + 1e-8)
    print(books_sub['year_norm'])

    # fill missing author/publisher
    books_sub['author'] = books_sub.get('author', 'Unknown')
    books_sub['publisher'] = books_sub.get('publisher', 'Unknown')
    print(books_sub[0:2])
    return users_sub.reset_index(drop=True), books_sub.reset_index(drop=True), ratings.reset_index(drop=True), user_map, book_map


if __name__ == '__main__':
    u, b, r = load_book_crossing("data")
    us, bs, rs, um, bm = preprocess(u, b, r, min_user_ratings=1, min_book_ratings=1)
    print('Synthetic -> users:', us.shape, 'books:', bs.shape, 'ratings:', rs.shape)


# Book Recommendation System (HybridNCF)

This project implements a **hybrid neural collaborative filtering (HybridNCF) model** for personalized book recommendations using the [Book-Crossing dataset](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset).

The system combines **user embeddings, item embeddings, author embeddings, publisher embeddings, and numeric features** (user age and book year) to predict book ratings and generate recommendations.

---

## Table of Contents
- [How to run](#howtorun)
- [Features](#features)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system
```
### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```
### 3. Train the model
```bash
python train.py --data_dir 'path_to_data_dir'
```
### 4. Evaluate model
```bash
python evaluate_model.py
```
### 5. Show top recommendations for a user
```bash
python book_recommendations.py
```

---

## Features
- Predicts user ratings for books.
- Recommends top-N books a user is likely to enjoy.
- Combines collaborative filtering and content features (author, publisher, numeric).
- Evaluates performance using RMSE and Precision/Recall@K.
- Supports GPU acceleration with PyTorch.

---

## Data

Expect a `data/` folder with the following files:

- `books.csv` – Book information (`ISBN`, `Book-Author`, `Year-Of-Publication`, `Publisher`)
- `ratings.csv` – User ratings (`User-ID`, `ISBN`, `Book-Rating`)
- `users.csv` – User data (`User-ID`, `Age`)

The preprocessing step filters out users and books with too few ratings and normalizes numeric features.

---

## Preprocessing

The preprocessing script:

- Renames columns to standard names.
- Removes zero ratings.
- Filters users with less than `min_user_ratings` ratings and books with less than `min_book_ratings`.
- Maps original user/book IDs to contiguous indices.
- Normalizes book publication year.
- Handles missing authors, publishers, and user ages.

---

## Model

The **HybridNCF** model consists of:

- Embeddings:
  - `user_emb`, `item_emb`, `author_emb`, `pub_emb`
- Numeric features:
  - `user_age`, `book_year_norm`
- MLP layers:
  - Configurable fully-connected layers with ReLU and Dropout.
- Output:
  - Predicts a single rating for each user-item pair.

The project also provides:

- `RecDataset` – Custom PyTorch Dataset for ratings.
- `collate_fn` – Converts list-of-samples into batched tensors.
- Helper functions for building author and publisher maps.

---

## Training

Training is done using **MSE loss** and the **Adam optimizer**.  

- Dataset split: 80% train, 20% test.
- Batch size: 256
- Epochs: configurable (example uses 2 for demonstration)
- Model checkpoints are saved per epoch and final model saved to `models/model_final.pth`.

## Evaluation

- **RMSE** for rating prediction (lower is better).
- **Precision@K / Recall@K** for top-N recommendations.



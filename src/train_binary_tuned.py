import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src import model_utils, text_preprocessing
from config import PROCESSED_DATA_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for binary tuned training."""
    parser = argparse.ArgumentParser(description="Train and evaluate the tuned baseline model.")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model to disk.",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "binary_logreg_tuned.joblib"),
        help="Output path for the saved model.",
    )
    return parser.parse_args()


def main():
    """Execute tuned model training pipeline"""
    args = parse_args()
    reviews_df = pd.read_parquet(PROCESSED_DATA_DIR / "amazon_reviews_features.parquet")

    reviews_df = model_utils.drop_neutral_ratings(reviews_df)
    reviews_df["rating"] = reviews_df["rating"].apply(model_utils.classify_rating)

    reviews_df["tfidf_text"] = reviews_df["full_text"].apply(text_preprocessing.clean_text)
    train_df, val_df, test_df = model_utils.split_data(reviews_df)

    tuned_model = model_utils.train_tf_idf_model(
        train_df,
        estimator=LogisticRegression(max_iter=1000, class_weight="balanced", C=10),
        vectorizer=TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            min_df=3,
            ngram_range=(1, 2),
        ),
    )

    model_utils.test_model(tuned_model, val_df, split_name="Validation")
    model_utils.test_model(tuned_model, test_df, split_name="Test")

    if args.save_model:
        saved_path = model_utils.save_model(tuned_model, args.model_path)
        print(f"\nSaved model to: {saved_path}")

if __name__ == "__main__":
    main()

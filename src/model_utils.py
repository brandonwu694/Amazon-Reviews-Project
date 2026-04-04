import pandas as pd
import joblib
from pathlib import Path
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not used as model features."""
    df = df.drop(columns=["review_date", "date_of_experience", "country", "country_grouped", "review_title", "review_text", "full_text", "review_count"])
    return df


def classify_rating(rating: int) -> int:
    """Classify 1-2 star ratings as poor (0) and 4-5 star ratings as good (1)"""
    if rating in (4, 5):
        return 1
    if rating in (1, 2):
        return 0
    raise ValueError("Rating must be 1, 2, 4, or 5 for binary classification.")


def drop_neutral_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop neutral (3-star) ratings"""
    if "rating" not in df.columns:
        raise ValueError("DataFrame must contain a 'rating' column.")
    return df[df["rating"] != 3].copy()


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into training, testing, and validationfolds"""
    if "rating" not in df.columns:
        raise ValueError("DataFrame must contain a 'rating' column.")
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["rating"]
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.25,
        random_state=42,
        stratify=train_df["rating"]
    )

    return train_df, val_df, test_df


def train_tf_idf_model(train_df: pd.DataFrame, estimator: BaseEstimator | None = None) -> Pipeline:
    """Train a TF-IDF plus structured-feature classification pipeline."""
    required_columns = {"rating", "tfidf_text", "log_review_count", "text_length"}
    missing_columns = required_columns.difference(train_df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_str}")

    modeling_df = drop_non_feature_columns(train_df.copy())

    text_feature = "tfidf_text"
    numeric_features = ["log_review_count", "text_length"]
    country_features = [col for col in modeling_df.columns if col.startswith("country_")]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5, max_features=10000),
                text_feature,
            ),
            ("num", StandardScaler(), numeric_features),
            ("country", "passthrough", country_features),
        ]
    )

    y_tr = modeling_df["rating"]
    X_tr = modeling_df.drop(columns=["rating"])

    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")

    baseline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )

    baseline_model.fit(X_tr, y_tr)

    return baseline_model


def evaluation_metrics(y: ArrayLike, preds: ArrayLike, split_name: str = "Validation") -> None:
    """Display evaluation metrics for model results."""
    print(f"{split_name} accuracy:", accuracy_score(y, preds))
    print(f"{split_name} macro F1:", f1_score(y, preds, average="macro"))
    print(f"{split_name} weighted F1:", f1_score(y, preds, average="weighted"))

    print(f"\n{split_name} classification report:")
    print(classification_report(y, preds))

    print(f"\n{split_name} confusion matrix:")
    print(confusion_matrix(y, preds))


def test_model(model: Pipeline, test_df: pd.DataFrame, split_name: str = "Validation") -> None:
    """Process test dataframe and evaluate model performance"""
    test_df = drop_non_feature_columns(test_df)
    y_test = test_df["rating"]
    X_test = test_df.drop(columns=["rating"])

    test_preds = model.predict(X_test)

    evaluation_metrics(y_test, test_preds, split_name)


def save_model(model: Pipeline, model_path: str | Path) -> Path:
    """Save a fitted model pipeline to disk."""
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path

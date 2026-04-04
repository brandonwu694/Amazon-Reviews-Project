import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not used as model features."""
    df = df.drop(columns=["review_date", "date_of_experience", "country", "country_grouped", "review_title", "review_text", "full_text", "review_count"])
    return df


def evaluation_metrics(y: ArrayLike, preds: ArrayLike) -> None:
    """Display evalutation metrics for model results"""
    print("Validation accuracy:", accuracy_score(y, preds))
    print("Validation macro F1:", f1_score(y, preds, average="macro"))
    print("Validation weighted F1:", f1_score(y, preds, average="weighted"))

    print("\nClassification report:")
    print(classification_report(y, preds))

    print("\nConfusion matrix:")
    print(confusion_matrix(y, preds))

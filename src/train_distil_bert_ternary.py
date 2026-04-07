import sys 
import argparse
from pathlib import Path 
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer

from config import BERT_DIR, MODELS_DIR


CHECKPOINT_DIR = MODELS_DIR / "distilbert_ternary_results"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for DistilBERT ternary classification training."""
    parser = argparse.ArgumentParser(description="Train and evaluate the DistilBERT ternary classifier.")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained DistilBERT model and tokenizer to disk.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODELS_DIR / "distilbert_ternary_classifier"),
        help="Output directory for the saved DistilBERT model.",
    )
    return parser.parse_args()


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into training, testing, and validation folds"""
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


def to_hf_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Dataset, Dataset, Dataset]:
    """Convert train, validation, and test folds into Hugging Face datasets"""
    # Conversion to Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    return train_ds, val_ds, test_ds


def tokenize_datasets(
        train_ds: Dataset, 
        val_ds: Dataset, 
        test_ds: Dataset, 
        tokenizer: DistilBertTokenizerFast) -> tuple[Dataset, Dataset, Dataset]:
    """Tokenize the text column for train, validation, and test folds"""
    train_ds = train_ds.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    val_ds = val_ds.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    test_ds = test_ds.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    return train_ds, val_ds, test_ds


def prepare_torch_datasets(train_ds: Dataset, val_ds: Dataset, test_ds: Dataset) -> tuple[Dataset, Dataset, Dataset]:
    """Remove raw text and format datasets for PyTorch training"""
    train_ds = train_ds.remove_columns(["full_text"])
    val_ds = val_ds.remove_columns(["full_text"])
    test_ds = test_ds.remove_columns(["full_text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    return train_ds, val_ds, test_ds


def train_distil_bert_model(
        train_ds: Dataset,
        val_ds: Dataset,
        model: DistilBertForSequenceClassification | None = None,
        training_args: TrainingArguments | None = None,
        trainer: Trainer | None = None) -> Trainer:
    """Train DistilBert model"""
    if model is None:
        # Load DistilBERT model
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=3
        )

    if training_args is None:
        training_args = TrainingArguments(
            output_dir=str(CHECKPOINT_DIR),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none"
        )

    if trainer is None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )

    trainer.train()
    return trainer


def evaluate_bert_model(trainer: Trainer, test_ds: Dataset, val_ds: Dataset) -> None:
    """Evaluate ternary DistilBERT performance with aggregate and per-class metrics."""
    val_results = trainer.predict(val_ds)
    test_results = trainer.predict(test_ds)

    print("Validation metrics:", val_results.metrics)
    print("Test metrics:", test_results.metrics)
    val_preds = np.argmax(val_results.predictions, axis=1)
    test_preds = np.argmax(test_results.predictions, axis=1)

    print("\nValidation classification report:")
    print(classification_report(val_results.label_ids, val_preds))
    print("Validation confusion matrix:")
    print(confusion_matrix(val_results.label_ids, val_preds))

    print("\nTest classification report:")
    print(classification_report(test_results.label_ids, test_preds))
    print("Test confusion matrix:")
    print(confusion_matrix(test_results.label_ids, test_preds))


def tokenize_function(examples: dict[str, list[str]], tokenizer: DistilBertTokenizerFast) -> Any:
    """Tokenize a batch of review texts for DistilBERT input."""
    return tokenizer(
        examples["full_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


def compute_metrics(eval_pred: Any) -> Any:
    """Compute macro-averaged metrics for ternary classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """Execute DistilBERT ternary classification training pipeline"""
    args = parse_args()

    reviews_df = pd.read_parquet(BERT_DIR / "amazon_reviews_bert_ternary.parquet")
    # Split data into training, validation, and test folds
    train_df, val_df, test_df = split_data(reviews_df)

    # Rename rating columns to labels
    train_df = train_df.rename(columns={"rating": "labels"})
    val_df = val_df.rename(columns={"rating": "labels"})
    test_df = test_df.rename(columns={"rating": "labels"})
    # Conversion to Hugging Face datasets
    train_ds, val_ds, test_ds = to_hf_datasets(train_df, val_df, test_df)

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    # Take text column from each batch, tokenize, and create tokenized outputs input_ds and attention_mask
    train_ds, val_ds, test_ds = tokenize_datasets(train_ds, val_ds, test_ds, tokenizer)
    # Remove raw text column and format for PyTorch
    train_ds, val_ds, test_ds = prepare_torch_datasets(train_ds, val_ds, test_ds)
    # Set training arguments
    trainer = train_distil_bert_model(train_ds, val_ds)

    # Evaluate model 
    evaluate_bert_model(trainer, test_ds, val_ds)

    if args.save_model:
        save_dir = Path(args.model_dir)
        trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"\nSaved model to: {save_dir}")


if __name__ == "__main__":
    main()

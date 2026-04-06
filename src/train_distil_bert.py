# pyright: basic

import sys 
from pathlib import Path 
sys.path.append(str(Path().resolve().parents[0]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from typing import Any
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer

from config import BERT_DIR, MODELS_DIR


CHECKPOINT_DIR = MODELS_DIR / "distilbert_results"


def tokenize_function(examples: dict[str, list[str]], tokenizer: DistilBertTokenizerFast) -> Any:
    """Tokenize a batch of review texts for DistilBERT input."""
    return tokenizer(
        examples["full_text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )


def compute_metrics(eval_pred: Any) -> Any:
    """Compute binary classification metrics from model logits and labels."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """Execute DistilBERT training pipeline"""
    reviews_df = pd.read_parquet(BERT_DIR / "amazon_reviews_bert.parquet")

    # Split data into training, validation, and test folds
    train_df, test_df = train_test_split(
        reviews_df,
        test_size=0.2,
        random_state=42,
        stratify=reviews_df["rating"]
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.25,
        random_state=42,
        stratify=train_df["rating"]
    )

    # Conversion to Hugging Face datasets
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Take text column from each batch, tokenize, and create tokenized outputs input_ds and attention_mask
    train_ds = train_ds.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    val_ds = val_ds.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    test_ds = test_ds.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    # Remove raw text column and format for PyTorch
    train_ds = train_ds.remove_columns(["full_text"])
    val_ds = val_ds.remove_columns(["full_text"])
    test_ds = test_ds.remove_columns(["full_text"])

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    # Load DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    # Set training arguments
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

    # Create trainer, train, and evaluate
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_results = trainer.predict(val_ds)
    test_results = trainer.predict(test_ds)

    print("Validation metrics:", val_results.metrics)
    print("Test metrics:", test_results.metrics)


if __name__ == "__main__":
    main()

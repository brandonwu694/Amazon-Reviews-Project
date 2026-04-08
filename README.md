# Amazon Reviews Project

## Problem Formulation

The initial goal of this project was to predict Amazon product ratings (1–5 stars) using both user metadata (e.g., `review_count`, `country`) and textual review content. However, exploratory data analysis revealed a highly imbalanced label distribution: 1-star reviews accounted for approximately 61% of the dataset, while 3-star reviews represented only about 4%.

A baseline logistic regression model trained on the 5-class task achieved moderate performance, with an overall accuracy of approximately `0.70` but a much lower macro F1 score of `0.45`. This discrepancy indicated that the model performed unevenly across classes, particularly struggling to distinguish underrepresented and intermediate ratings such as 3-star reviews.

Given that the model more reliably captured extreme sentiment (e.g., clearly positive or negative reviews) than nuanced middle ratings, the problem was reformulated as a binary classification task. Ratings of 1–2 were grouped as poor, and ratings of 4–5 as good, with 3-star reviews excluded due to their ambiguity and low representation.

This reformulation significantly improved model performance, resulting in a more stable and reliable baseline.

## Data Access

The raw dataset is not stored in this repository. Download `Amazon_Reviews.csv` [here](https://www.kaggle.com/datasets/dongrelaxman/amazon-reviews-dataset?resource=download) and place it in:

`data/raw/Amazon_Reviews.csv`

After downloading the raw data, run the data preparation notebooks/scripts in order to reproduce the processed datasets used by the modeling pipelines.

`data/` directory structure: 

```text
data/
├── raw/
│   └── Amazon_Reviews.csv
├── processed/
│   ├── amazon_reviews_clean.parquet
│   └── amazon_reviews_features.parquet
└── bert/
    └── amazon_reviews_bert.parquet
```

## Tuned LR Shortcomings

The tuned binary logistic regression model is strong, but notebook `08_error_analysis.ipynb` shows a few clear limitations:

- It still makes a non-trivial number of high-confidence mistakes, which suggests some errors are model limitations rather than simple threshold issues.
- It struggles with mixed-sentiment reviews where positive and negative language appear in the same example.
- It misses contextual cues such as contrast and negation, for example reviews that begin positively and then describe a bad experience.
- Some longer reviews still show weaker class balance performance than the aggregate metrics suggest.
- Some rows contain weak text such as `Review text not found`, which limits what any text model can learn from that input.

These findings make a lightweight transformer comparison defensible, especially `DistilBERT`, since the remaining errors appear to be driven more by context and phrasing than by simple keyword presence.

## DistilBert Results and Shortcomings

The initial iteration of the lightweight transformer performed well on reviews with clear positive or negative sentiment, but struggled with more ambiguous inputs (e.g., “X is good, but Y is bad”), often relying on surface-level cues.

To improve performance on ambiguous reviews, the model was retrained on a ternary classification task: 1–2 stars as poor, 3 stars as neutral, and 4–5 stars as good. While the retrained model continued to perform well on clearly positive and negative reviews, it struggled to classify neutral reviews accurately.

The confusion matrix showed that neutral reviews were more often misclassified as negative than positive, suggesting a bias toward negative predictions when handling mixed sentiment, possibly due to stronger or more noticeable negative signals in these reviews.

Based on these findings, the problem was reformulated as a binary classification task, grouping 1–3 star ratings as non-positive and 4–5 star ratings as positive.

### Binary Classifier Test Performance 

**Note:** The binary classifier was trained and evaluated using only 1, 2, 4, and 5-star reviews. 3-star (neutral) reviews were excluded from this experiment.

| Metric            | Value  |
| ----------------- | ------ |
| **Loss**          | 0.1247 |
| **Accuracy**      | 0.9724 |
| **Precision**     | 0.9549 |
| **Recall**        | 0.9491 |
| **F1-score**      | 0.9520 |
| **Runtime (s)**   | 103.37 |
| **Samples / sec** | 38.98  |
| **Steps / sec**   | 2.44   |

### Ternary Classifier Test Performance 

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0 (Poor)         | 0.96      | 0.97   | 0.96     | 2869    |
| 1 (Neutral)      | 0.29      | 0.21   | 0.24     | 177     |
| 2 (Good)         | 0.91      | 0.93   | 0.92     | 1160    |
| **Accuracy**     |           |        | **0.93** | 4206    |
| **Macro Avg**    | 0.72      | 0.70   | 0.71     | 4206    |
| **Weighted Avg** | 0.92      | 0.93   | 0.92     | 4206    |

### Ternary Classifier Confusion Matrix

| Actual \ Predicted | 0 (Poor) | 1 (Neutral) | 2 (Good) |
| ------------------ | -------- | ----------- | -------- |
| **0 (Poor)**       | 2774     | 50          | 45       |
| **1 (Neutral)**    | 83       | 37          | 57       |
| **2 (Good)**       | 41       | 39          | 1080     |

## Model Shortcomings and Future Steps

Both variations of the trained `DistilBert` models performed well when tested on both the validation and testing sets. However, when the model was tasked to classify unseen, more ambiguous reviews, it often misclassified them with high confidence. To make the model more generalizable to unseen data, it can be trained on a larger volume of data, with a more balanced distribution of ratings. Furthermore, converting the problem from 5-class classification into a binary classification problem may have unintentionally made it harder for the model to understand milder sentiments in reviews. Ultimately, training the model on a larger and more diverse dataset and reformulating the problem to better model neutral reviews are the two highest impact changes to improve the model's performance. 

## Run the Binary Baseline Pipeline

Run the training pipeline from the repository root:

```bash
amazon/bin/python src/train_binary_baseline.py
```

This will:
- load the processed review dataset
- drop 3-star reviews
- convert ratings into a binary target
- train the TF-IDF + logistic regression baseline
- print validation and test metrics to the CLI

To train the model and save the fitted pipeline to disk:

```bash
amazon/bin/python src/train_binary_baseline.py --save-model
```

By default, the saved model is written to:

```text
models/binary_logreg_baseline.joblib
```

To save the model to a custom path:

```bash
amazon/bin/python src/train_binary_baseline.py --save-model --model-path models/custom_binary_baseline.joblib
```

## Run the Tuned Binary Pipeline

Run the tuned logistic regression pipeline from the repository root:

```bash
amazon/bin/python src/train_binary_tuned.py
```

This will:
- load the processed review dataset
- drop 3-star reviews
- convert ratings into a binary target
- train the tuned TF-IDF + logistic regression model
- print validation and test metrics to the CLI

To train the tuned model and save the fitted pipeline to disk:

```bash
amazon/bin/python src/train_binary_tuned.py --save-model
```

By default, the saved tuned model is written to:

```text
models/binary_logreg_tuned.joblib
```

To save the tuned model to a custom path:

```bash
amazon/bin/python src/train_binary_tuned.py --save-model --model-path models/custom_binary_tuned.joblib
```

## Run the Binary DistilBERT Pipeline

Run the binary DistilBERT training pipeline from the repository root:

```bash
amazon/bin/python src/train_distil_bert.py
```

This will:
- load the BERT-ready binary review dataset
- split the data into training, validation, and test folds
- tokenize review text with `distilbert-base-uncased`
- fine-tune a DistilBERT binary classifier
- print validation and test metrics to the CLI

To train the model and save the final model and tokenizer:

```bash
amazon/bin/python src/train_distil_bert.py --save-model
```

By default, the saved DistilBERT artifacts are written to:

```text
models/distilbert_binary_classifier/
```

Training checkpoints are written to:

```text
models/distilbert_results/
```

## Run the Ternary DistilBERT Pipeline

Run the ternary DistilBERT training pipeline from the repository root:

```bash
amazon/bin/python src/train_distil_bert_ternary.py
```

This will:
- load the BERT-ready ternary review dataset
- split the data into training, validation, and test folds
- tokenize review text with `distilbert-base-uncased`
- fine-tune a DistilBERT ternary classifier
- print validation and test metrics to the CLI
- print classification reports and confusion matrices for validation and test folds

To train the ternary model and save the final model and tokenizer:

```bash
amazon/bin/python src/train_distil_bert_ternary.py --save-model
```

By default, the saved ternary DistilBERT artifacts are written to:

```text
models/distilbert_ternary_classifier/
```

Training checkpoints are written to:

```text
models/distilbert_ternary_results/
```

## Deploy on Render

This repository includes a [render.yaml](/Users/brandonwu/Documents/amazon_reviews_project/render.yaml) for deploying the FastAPI app as a Render web service.

The API starts with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

To deploy on Render:

1. Push this repository to GitHub.
2. In Render, create a new Blueprint instance or Web Service from the repository.
3. Confirm the service uses `pip install -r requirements.txt` as the build command.
4. Confirm the start command is `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

The configured health check endpoint is:

```text
/health
```

Important deployment note:
- The binary DistilBERT model files under `models/distilbert_binary_classifier/` must remain in the repository, since the API loads them from local disk at startup.

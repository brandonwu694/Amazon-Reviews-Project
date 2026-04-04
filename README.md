# Amazon Reviews Project

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

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

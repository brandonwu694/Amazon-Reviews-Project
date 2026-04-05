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

## Tuned LR Shortcomings

The tuned binary logistic regression model is strong, but notebook `08_error_analysis.ipynb` shows a few clear limitations:

- It still makes a non-trivial number of high-confidence mistakes, which suggests some errors are model limitations rather than simple threshold issues.
- It struggles with mixed-sentiment reviews where positive and negative language appear in the same example.
- It misses contextual cues such as contrast and negation, for example reviews that begin positively and then describe a bad experience.
- Some longer reviews still show weaker class balance performance than the aggregate metrics suggest.
- Some rows contain weak text such as `Review text not found`, which limits what any text model can learn from that input.

These findings make a lightweight transformer comparison defensible, especially `DistilBERT`, since the remaining errors appear to be driven more by context and phrasing than by simple keyword presence.

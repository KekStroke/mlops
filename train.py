"""
Train a text spam classifier (Logistic Regression or Naive Bayes) on SMS Spam data.

Usage:
    python train.py --dataset-id <DATASET_ID>
    python train.py --dataset-id <DATASET_ID> --model_type nb --ngram_max 2
    python train.py --dataset-id <DATASET_ID> --queue default --model_type logreg --max_features 5000
"""

import argparse
import glob
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, OutputModel, Task
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_NAME = "Course MLOps"
MODEL_DIR = "model_output"

CLEARML_OUTPUT_URI = os.getenv(
    "CLEARML_OUTPUT_URI",
    "https://192.168.0.105:8081"
)

# ── argparse ─────────────────────────────────────────────────────────┐


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train spam classifier on ClearML dataset.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="ClearML dataset id",
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="students",
        help='ClearML queue name for remote execution (default: "students")',
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="logreg",
        choices=["logreg", "nb"],
        help='Classifier type (default: "logreg")',
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=3000,
        help="Max TF-IDF features (default: 3000)",
    )
    parser.add_argument(
        "--ngram_max",
        type=int,
        default=1,
        help="Upper bound of n-gram range (default: 1)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for split (default: 42)",
    )
    return parser.parse_args()


# ── data loading ──────────────────────────────────────────────────────|


def load_dataset_from_clearml(dataset_id: str) -> pd.DataFrame:
    """Download ClearML dataset locally and read the first CSV file found."""
    print(f"Fetching ClearML dataset {dataset_id}...")
    dataset = Dataset.get(dataset_id=dataset_id)
    local_path = dataset.get_local_copy()

    csv_files = glob.glob(os.path.join(local_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found inside dataset directory: {local_path}"
        )

    csv_path = csv_files[0]
    print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"CSV must contain 'label' and 'text' columns. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["label"] = df["label"].astype(str)
    df["text"] = df["text"].astype(str)

    print(
        f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts().to_string()}"
    )
    return df


# ── pipeline ──────────────────────────────────────────────────────────|


def build_pipeline(
    model_type: str,
    max_features: int,
    ngram_max: int,
) -> Pipeline:
    """Build a sklearn Pipeline: TfidfVectorizer -> Classifier."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        stop_words="english",
    )

    if model_type == "logreg":
        classifier = LogisticRegression(max_iter=1000)
    elif model_type == "nb":
        classifier = MultinomialNB()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )


# ── train & evaluate ─────────────────────────────────────────────────|


def train_and_evaluate(
    df: pd.DataFrame,
    pipeline: Pipeline,
    test_size: float,
    random_state: int,
) -> dict:
    """Split data, train pipeline, return metrics dict."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 macro : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    return {"accuracy": acc, "f1_macro": f1, "y_test": y_test, "y_pred": y_pred}


# ── confusion matrix ─────────────────────────────────────────────────|


def log_confusion_matrix_image(
    y_test: pd.Series,
    y_pred: pd.Series,
    logger,
) -> None:
    """Build confusion matrix, log as image to ClearML."""
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)

    tick_marks = range(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()

    logger.report_matplotlib_figure(
        title="Confusion Matrix",
        series="confusion_matrix",
        figure=fig,
        report_image=True,
    )
    plt.close(fig)
    print("Confusion matrix logged to ClearML.")


# ── save & register ──────────────────────────────────────────────────|


def save_and_register_model(
    task: Task,
    pipeline: Pipeline,
    accuracy: float,
    model_type: str,
) -> str:
    """Dump model to joblib, upload as artifact, register as OutputModel."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    task.upload_artifact(name="model_artifact", artifact_object=model_path)

    output_model = OutputModel(
        task=task,
        framework="scikit-learn",
        name=f"spam_clf_{model_type}",
    )
    output_model.update_weights(
        weights_filename=model_path, upload_uri=CLEARML_OUTPUT_URI
    )

    task.add_tags([model_type, f"accuracy={accuracy:.4f}"])
    output_model.tags = [model_type, "spam-classifier"]

    print(f"Model registered in ClearML (id={output_model.id}).")
    return model_path


# ── main ──────────────────────────────────────────────────────────────|


def main():
    args = parse_args()

    task = Task.init(
        project_name=PROJECT_NAME,
        task_name=f"train_{args.model_type}",
        task_type=Task.TaskTypes.training,
        output_uri=CLEARML_OUTPUT_URI,
        reuse_last_task_id=False,
    )
    task.connect(vars(args))

    # ── remote execution (no-op when running locally without queue) ───|
    task.execute_remotely(queue_name=args.queue)

    # 1  load data ─────────────────────────────────────────────────────|
    df = load_dataset_from_clearml(args.dataset_id)

    # 2  build pipeline ────────────────────────────────────────────────|
    pipeline = build_pipeline(
        model_type=args.model_type,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
    )

    # 3  train & evaluate ──────────────────────────────────────────────|
    metrics = train_and_evaluate(
        df=df,
        pipeline=pipeline,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # 4  log scalars ───────────────────────────────────────────────────|
    logger = task.get_logger()
    logger.report_single_value("accuracy", metrics["accuracy"])
    logger.report_single_value("f1_macro", metrics["f1_macro"])
    print("Metrics logged to ClearML.")

    # 5  confusion matrix ──────────────────────────────────────────────|
    log_confusion_matrix_image(metrics["y_test"], metrics["y_pred"], logger)

    # 6  save & register model ─────────────────────────────────────────|
    model_path = save_and_register_model(
        task=task,
        pipeline=pipeline,
        accuracy=metrics["accuracy"],
        model_type=args.model_type,
    )

    # 7  summary ───────────────────────────────────────────────────────|
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  accuracy       = {metrics['accuracy']:.4f}")
    print(f"  f1_macro       = {metrics['f1_macro']:.4f}")
    print(f"  model_path     = {model_path}")
    print(f"  clearml_task   = {task.id}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

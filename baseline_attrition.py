from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition 2.csv")
OUTPUT_DIR = Path("baseline_results")
RANDOM_STATE = 42


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(DATA_PATH)

    if "Attrition" not in data.columns:
        raise ValueError("Expected an 'Attrition' target column")

    y = data["Attrition"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Attrition must contain only 'Yes' or 'No'")

    # These columns are identifiers or constants and should not train the model.
    drop_columns = [
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
    x = data.drop(columns=[column for column in drop_columns if column in data.columns])
    return x, y.astype(int)


def build_model(x: pd.DataFrame) -> Pipeline:
    categorical_columns = x.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = x.select_dtypes(include=["number"]).columns.tolist()

    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), numerical_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model(x)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_probability = model.predict_proba(x_test)[:, 1]

    matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
        "pr_auc": float(average_precision_score(y_test, y_probability)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "matrix": matrix.tolist(),
        },
        "dataset": {
            "total_rows": int(len(y)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "attrition_rows": int(y.sum()),
            "attrition_rate": float(y.mean()),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    pd.DataFrame(
        [{"metric": name, "value": value} for name, value in metrics.items() if isinstance(value, float)]
    ).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Attrition", "Attrition"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Logistic Regression Baseline — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

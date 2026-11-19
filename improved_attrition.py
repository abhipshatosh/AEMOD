from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition 2.csv")
OUTPUT_DIR = Path("improved_results")
RANDOM_STATE = 42


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    result["PromotionDelay"] = (
        result["YearsAtCompany"] - result["YearsSinceLastPromotion"]
    ).clip(lower=0)

    job_level_income = result.groupby("JobLevel")["MonthlyIncome"].transform("mean")
    result["IncomeRatio"] = result["MonthlyIncome"] / job_level_income.replace(0, 1)

    result["TenureCategory"] = pd.cut(
        result["YearsAtCompany"],
        bins=[-1, 2, 5, 10, float("inf")],
        labels=["0-2 years", "3-5 years", "6-10 years", "10+ years"],
    )

    overtime_by_role = result.groupby("JobRole")["OverTime"].transform(
        lambda values: (values == "Yes").mean()
    )
    result["OvertimePercentage"] = overtime_by_role

    return result


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(DATA_PATH)
    if "Attrition" not in data.columns:
        raise ValueError("Expected an 'Attrition' target column")

    y = data["Attrition"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Attrition must contain only 'Yes' or 'No'")

    drop_columns = [
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours",
    ]
    x = data.drop(columns=[column for column in drop_columns if column in data.columns])
    x = engineer_features(x)
    return x, y.astype(int)


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical_columns = x.select_dtypes(include=["number", "bool"]).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_searches(preprocessor: ColumnTransformer):
    logistic = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    gradient_boosting = ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]
    )

    xgboost = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    searches = {
        "logistic_regression": (
            logistic,
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
        ),
        "random_forest": (
            random_forest,
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [6, 12, None],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__max_features": ["sqrt", 0.7],
            },
        ),
        "gradient_boosting": (
            gradient_boosting,
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
                "classifier__subsample": [0.8, 1.0],
            },
        ),
        "xgboost": (
            xgboost,
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
                "classifier__scale_pos_weight": [1, 3, 5],
            },
        ),
    }
    return searches


def calculate_metrics(y_test: pd.Series, y_pred, y_probability) -> dict:
    matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = matrix.ravel()
    return {
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
    }


def save_results(results: dict, y_test: pd.Series, y_pred, best_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "model_comparison.json").write_text(json.dumps(results, indent=2) + "\n")

    rows = []
    for name, result in results.items():
        rows.append({"model": name, **{key: value for key, value in result.items() if isinstance(value, float)}})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["No Attrition", "Attrition"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"{best_name} — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    x, y = load_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    preprocessor = make_preprocessor(x)
    searches = build_searches(preprocessor)

    results = {}
    fitted_models = {}

    for name, (pipeline, parameter_grid) in searches.items():
        print(f"Tuning {name}...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)

        y_pred = search.predict(x_test)
        y_probability = search.predict_proba(x_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred, y_probability)
        metrics["best_cv_f1"] = float(search.best_score_)
        metrics["best_parameters"] = search.best_params_

        results[name] = metrics
        fitted_models[name] = search
        print(json.dumps({name: metrics}, indent=2))

    best_name = max(results, key=lambda name: results[name]["f1_score"])
    best_search = fitted_models[best_name]
    best_pred = best_search.predict(x_test)

    save_results(results, y_test, best_pred, best_name)

    print("\nBest model:", best_name)
    print(json.dumps(results[best_name], indent=2))
    print(f"\nResults saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

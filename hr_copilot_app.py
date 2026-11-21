from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("data/employee_attrition_dataset_10000.csv")
RANDOM_STATE = 42


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["AttritionFlag"] = (data["Attrition"] == "Yes").astype(int)
    return data


@st.cache_resource
def train_model(data: pd.DataFrame):
    y = data["AttritionFlag"]
    x = data.drop(columns=["Attrition", "AttritionFlag", "Employee_ID"])

    categorical = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical = x.select_dtypes(include=["number", "bool"]).columns.tolist()

    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numerical),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, C=1),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []
    fitted = {}
    for name, classifier in models.items():
        model = Pipeline([("preprocess", preprocessing), ("model", classifier)])
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        fitted[name] = model
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
                "PR-AUC": average_precision_score(y_test, probabilities),
            }
        )

    comparison = pd.DataFrame(results)
    best_name = comparison.sort_values(["F1", "PR-AUC", "ROC-AUC"], ascending=False).iloc[0]["Model"]
    best_model = fitted[best_name]

    permutation = permutation_importance(
        best_model,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame({"Feature": x.columns, "Importance": permutation.importances_mean})
        .sort_values("Importance", ascending=False)
        .head(12)
    )

    return {
        "comparison": comparison,
        "best_name": best_name,
        "best_model": best_model,
        "importance": importance,
        "x_test": x_test,
        "y_test": y_test,
    }


def answer_query(question: str, data: pd.DataFrame, trained: dict) -> str:
    question = question.lower()

    if "overall" in question or "dataset" in question:
        rate = data["AttritionFlag"].mean() * 100
        return f"The dataset contains {len(data):,} employees. The overall attrition rate is {rate:.1f}% ({int(data['AttritionFlag'].sum()):,} leavers)."

    if "department" in question:
        summary = data.groupby("Department")["AttritionFlag"].mean().mul(100).sort_values(ascending=False)
        top = summary.index[0]
        return f"{top} has the highest attrition rate at {summary.iloc[0]:.1f}%. Department rates: " + ", ".join(f"{department} {value:.1f}%" for department, value in summary.items()) + "."

    if "overtime" in question:
        summary = data.groupby("Overtime")["AttritionFlag"].mean().mul(100)
        return "Overtime attrition rates: " + ", ".join(f"{value}: {summary[value]:.1f}%" for value in summary.index) + "."

    if "driver" in question or "factor" in question:
        top = trained["importance"].head(5)
        return "The highest permutation-importance attrition drivers are: " + ", ".join(top["Feature"].tolist()) + ". These indicate where changing or investigating a feature has the strongest relationship with model risk."

    if "best model" in question or "which model" in question:
        row = trained["comparison"].loc[trained["comparison"]["Model"] == trained["best_name"]].iloc[0]
        return f"The best model is {trained['best_name']}. It has F1 {row['F1']:.3f}, recall {row['Recall']:.3f}, ROC-AUC {row['ROC-AUC']:.3f}, and PR-AUC {row['PR-AUC']:.3f}."

    if "high risk" in question or "top employee" in question:
        probabilities = trained["best_model"].predict_proba(trained["x_test"])[:, 1]
        return f"The held-out test set contains {sum(probabilities >= 0.5)} employees above a 0.50 risk threshold. The highest predicted risk is {probabilities.max():.1%}."

    return "I can answer questions about overall attrition, department risk, overtime, model performance, key drivers, or high-risk employees. Try: 'Which department has the highest attrition?'"


def main() -> None:
    st.set_page_config(page_title="HR Attrition Copilot", layout="wide")
    st.title("HR Attrition Copilot")
    st.caption("Predict employee attrition, explain contributors, and explore retention risk through analytics.")

    data = load_data()

    with st.spinner("Training predictive models..."):
        trained = train_model(data)

    overview, analytics, copilot, employee = st.tabs(
        ["Overview", "Analytics", "Ask the Copilot", "Employee Risk"]
    )

    with overview:
        left, middle, right = st.columns(3)
        left.metric("Employees", f"{len(data):,}")
        middle.metric("Attrition Rate", f"{data['AttritionFlag'].mean():.1%}")
        right.metric("Best Model", trained["best_name"])

        rate_by_department = data.groupby("Department")["AttritionFlag"].mean().mul(100).reset_index()
        st.plotly_chart(px.bar(rate_by_department, x="Department", y="AttritionFlag", title="Attrition Rate by Department", labels={"AttritionFlag": "Attrition rate (%)"}), use_container_width=True)

    with analytics:
        st.subheader("Model Comparison")
        st.dataframe(trained["comparison"].round(4), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(trained["importance"].sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Top Attrition Drivers (Permutation Importance)"), use_container_width=True)
        with col2:
            overtime = data.groupby("Overtime")["AttritionFlag"].mean().mul(100).reset_index()
            st.plotly_chart(px.bar(overtime, x="Overtime", y="AttritionFlag", title="Attrition by Overtime", labels={"AttritionFlag": "Attrition rate (%)"}), use_container_width=True)

    with copilot:
        st.subheader("Ask a question")
        question = st.text_input("Question", placeholder="Which department has the highest attrition?")
        for suggestion in [
            "What is the overall attrition rate?",
            "Which department has the highest attrition?",
            "How does overtime affect attrition?",
            "What are the key attrition drivers?",
            "Which model is best?",
        ]:
            st.button(suggestion, on_click=lambda q=suggestion: st.session_state.update({"copilot_question": q}))

        active_question = st.session_state.pop("copilot_question", question)
        if active_question:
            st.info(answer_query(active_question, data, trained))

    with employee:
        st.subheader("Held-out Employee Risk")
        probabilities = trained["best_model"].predict_proba(trained["x_test"])[:, 1]
        risk = trained["x_test"].copy()
        risk["AttritionProbability"] = probabilities
        risk["RiskBand"] = pd.cut(
            risk["AttritionProbability"],
            bins=[-1, 0.33, 0.66, 1.01],
            labels=["Low", "Medium", "High"],
        )
        risk = risk.sort_values("AttritionProbability", ascending=False)

        threshold = st.slider("High-risk threshold", 0.05, 0.95, 0.50, 0.05)
        high_risk = risk[risk["AttritionProbability"] >= threshold]
        st.metric("Employees Above Threshold", len(high_risk))
        st.dataframe(high_risk.head(100), use_container_width=True)


if __name__ == "__main__":
    main()

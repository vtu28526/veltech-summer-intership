from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "disease_outbreak_sample.csv"
OUTPUT_DIR = BASE_DIR / "outbreak_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_PATH = OUTPUT_DIR / "predict_vs_actual_plot.png"
SPLIT_CSV_PATH = OUTPUT_DIR / "train_test_split_explorer.csv"
FEATURE_RMSE_CSV_PATH = OUTPUT_DIR / "feature_comparison_rmse.csv"
MODEL_PATH = OUTPUT_DIR / "outbreak_risk_model_pickle.pkl"


RISK_OUTPUTS_STYLE = {
    "figure.facecolor": "#0b1220",
    "axes.facecolor": "#0f1a2b",
    "axes.edgecolor": "#334a6b",
    "axes.labelcolor": "#cfe3ff",
    "xtick.color": "#cfe3ff",
    "ytick.color": "#cfe3ff",
    "text.color": "#cfe3ff",
    "grid.color": "#2a3a57",
    "grid.alpha": 0.55,
    "legend.frameon": False,
}


def set_custom_plot_style() -> None:
    plt.rcParams.update(RISK_OUTPUTS_STYLE)


def add_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vaccination_risk = (100 - df["vaccination_rate"]) / 100
    density_risk = df["population_density"] / df["population_density"].max()
    sanitation_risk = (100 - df["sanitation_score"]) / 100

    df["risk_index"] = (
        0.40 * vaccination_risk + 0.35 * density_risk + 0.25 * sanitation_risk
    ).round(3)
    return df


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Safety imputation (scripts used similar approach but not always identical)
    if "vaccination_rate" in df.columns:
        df["vaccination_rate"] = df["vaccination_rate"].fillna(df["vaccination_rate"].median())
    if "sanitation_score" in df.columns:
        df["sanitation_score"] = df["sanitation_score"].fillna(df["sanitation_score"].median())
    if "cases_last_week" in df.columns:
        df["cases_last_week"] = df["cases_last_week"].fillna(df["cases_last_week"].median())

    return add_risk_index(df)


def your_first_regression(df: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame]:
    numeric_df = df.select_dtypes(include="number").copy()
    numeric_df = numeric_df.fillna(numeric_df.median())

    x = numeric_df.drop(columns=["cases_last_week"])
    y = numeric_df["cases_last_week"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("YOUR FIRST REGRESSION")
    print("Numeric columns used:")
    print(list(x.columns))
    print(f"Train rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.3f}")

    result = pd.DataFrame(
        {"actual_cases": y_test.values, "predicted_cases": predictions.round(2)}
    )
    print("Actual vs Predicted:")
    print(result.to_string(index=False))

    return model, result


def predict_vs_actual_plot(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number").copy()
    numeric_df = numeric_df.fillna(numeric_df.median())

    x = numeric_df.drop(columns=["cases_last_week"])
    y = numeric_df["cases_last_week"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("PREDICT VS ACTUAL PLOT")
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.3f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.9, s=55, color="#4ea1ff", edgecolors="#0b1220")
    plt.xlabel("Actual Cases")
    plt.ylabel("Predicted Cases")
    plt.title("Predicted vs Actual Outbreak Cases")
    plt.grid(True)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="#ff3d71", linewidth=2)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180, facecolor=plt.gcf().get_facecolor())
    plt.close()

    print(f"Plot saved to: {PLOT_PATH}")


def train_test_split_explorer(df: pd.DataFrame, test_sizes=None) -> pd.DataFrame:
    if test_sizes is None:
        test_sizes = [0.2, 0.25, 0.3, 0.35, 0.4]

    features = [
        "vaccination_rate",
        "population_density",
        "sanitation_score",
        "risk_index",
    ]

    x = df[features]
    y = df["cases_last_week"]

    results = []

    for test_size in test_sizes:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=42
        )

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        results.append(
            {
                "test_size": test_size,
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "mae": round(mean_absolute_error(y_test, predictions), 2),
                "r2_score": round(r2_score(y_test, predictions), 3),
            }
        )

    return pd.DataFrame(results)


def feature_comparison_rmse(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number").copy()
    numeric_df = numeric_df.fillna(numeric_df.median())

    target = "cases_last_week"
    features = [c for c in numeric_df.columns if c != target]

    results = []

    for feature in features:
        x = numeric_df[[feature]]
        y = numeric_df[target]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42
        )

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rmse = mean_squared_error(y_test, predictions) ** 0.5

        results.append({"feature": feature, "rmse": round(rmse, 2)})

    return pd.DataFrame(results).sort_values("rmse")


def save_and_load_model_pickle(df: pd.DataFrame) -> None:
    features = [
        "vaccination_rate",
        "population_density",
        "sanitation_score",
        "risk_index",
    ]

    x = df[features]
    y = df["risk_class"]

    # Avoid stratify errors on small datasets: compute a test size that leaves
    # at least one sample per class.
    classes = y.unique()
    n_classes = len(classes)
    raw_test_size = 0.25
    # need test_size * n_samples >= n_classes
    test_size = max(raw_test_size, n_classes / max(len(y), 1))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("SAVE YOUR MODEL (PICKLE)")
    print(classification_report(y_test, y_pred, zero_division=0))

    model_bundle = {
        "model": model,
        "features": features,
        "classes": sorted(y.unique()),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"Model saved to: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        loaded_bundle = pickle.load(f)

    sample_input = pd.DataFrame(
        [
            {
                "vaccination_rate": 55,
                "population_density": 9000,
                "sanitation_score": 45,
                "risk_index": 0.536,
            }
        ]
    )

    loaded_model = loaded_bundle["model"]
    prediction = loaded_model.predict(sample_input[loaded_bundle["features"]])[0]
    probabilities = loaded_model.predict_proba(sample_input[loaded_bundle["features"]])[0]

    print("Loaded model test prediction")
    print(f"Predicted risk class: {prediction}")
    print(f"Prediction probability: {max(probabilities):.3f}")


def main() -> None:
    set_custom_plot_style()
    df = load_dataset()

    # 1) Regression explorer (first regression)
    your_first_regression(df)

    # 2) Predict vs actual plot
    predict_vs_actual_plot(df)

    # 3) Train test split explorer
    split_results = train_test_split_explorer(df, test_sizes=[0.2, 0.25, 0.3, 0.35, 0.4])
    print("TRAIN TEST SPLIT EXPLORER")
    print(split_results.to_string(index=False))
    split_results.to_csv(SPLIT_CSV_PATH, index=False)
    print(f"Saved results to: {SPLIT_CSV_PATH}")

    # 4) Feature comparison (RMSE)
    comparison = feature_comparison_rmse(df)
    print("FEATURE COMPARISON - RMSE")
    print(comparison.to_string(index=False))
    best_feature = comparison.iloc[0]
    print(f"Best single feature: {best_feature['feature']} with RMSE {best_feature['rmse']}")
    comparison.to_csv(FEATURE_RMSE_CSV_PATH, index=False)
    print(f"Saved results to: {FEATURE_RMSE_CSV_PATH}")

    # 5) Save model pickle (classification)
    save_and_load_model_pickle(df)

    print("\nDONE: Outputs are saved in outbreak_outputs/.")


if __name__ == "__main__":
    main()


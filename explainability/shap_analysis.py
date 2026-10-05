"""
SolarSentinel-X
Phase 4 - SHAP Explainability

Production Version
"""

from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest"
    / "saved_models"
    / "random_forest_model.pkl"
)

TRAIN_DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "X_train.csv"
)

TEST_DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "X_test.csv"
)

ENCODER_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "health_encoder.pkl"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "shap"
)

LOCAL_RESULTS_DIR = RESULTS_DIR / "local"


# ==========================================================
# Create Output Folder
# ==========================================================

def create_output_directory():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Load Random Forest Model
# ==========================================================

def load_model():

    print("=" * 60)
    print("Loading Random Forest Model...")
    print("=" * 60)

    model = joblib.load(MODEL_PATH)

    print("✓ Model Loaded Successfully\n")

    return model


# ==========================================================
# Load Training Dataset
# ==========================================================

def load_training_data():

    print("Loading Training Dataset...")

    X_train = pd.read_csv(TRAIN_DATA_PATH)

    print("✓ Dataset Loaded Successfully")

    print(f"Samples  : {X_train.shape[0]}")
    print(f"Features : {X_train.shape[1]}\n")

    return X_train

# ==========================================================
# Load Test Dataset
# ==========================================================

def load_test_data():

    print("Loading Test Dataset...")

    X_test = pd.read_csv(TEST_DATA_PATH)

    print("✓ Test Dataset Loaded Successfully")

    print(f"Samples  : {X_test.shape[0]}")
    print(f"Features : {X_test.shape[1]}\n")

    return X_test


# ==========================================================
# Create SHAP Explainer
# ==========================================================

def create_explainer(model):

    print("Creating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("✓ TreeExplainer Created\n")

    return explainer


# ==========================================================
# Compute SHAP Values
# ==========================================================

def compute_shap_values(explainer, X_train):

    print("Computing SHAP Values...")

    shap_values = explainer(X_train)

    print("✓ SHAP Values Computed\n")

    print(f"SHAP Object Type : {type(shap_values)}")
    print(f"SHAP Shape       : {shap_values.values.shape}")
    print(f"Feature Shape    : {X_train.shape}\n")

    return shap_values

# ==========================================================
# Save SHAP Values
# ==========================================================

def save_shap_values(shap_values):

    print("Saving SHAP Values...")

    shap_path = RESULTS_DIR / "shap_values.npy"

    feature_path = RESULTS_DIR / "feature_names.csv"

    import numpy as np

    np.save(
        shap_path,
        shap_values.values
    )

    pd.DataFrame({
        "Feature": shap_values.feature_names
    }).to_csv(
        feature_path,
        index=False
    )

    print("✓ Saved : shap_values.npy")
    print("✓ Saved : feature_names.csv\n")


# ==========================================================
# Detect Classes
# ==========================================================

def detect_classes(model):

    class_labels = model.classes_

    print("Detected Classes:")

    for label in class_labels:
        print(f"• {label}")

    print()

    return class_labels


# ==========================================================
# Main
# ==========================================================

def main():

    create_output_directory()

    model = load_model()
    
    health_encoder = load_health_encoder()

    X_train = load_training_data()

    X_test = load_test_data()

    explainer = create_explainer(model)

    shap_values = compute_shap_values(
        explainer,
        X_test
    )
    
    save_shap_values(
        shap_values
    )

    class_labels = detect_classes(model)

    generate_summary_plots(
        shap_values,
        class_labels,
        health_encoder
    )

    generate_global_bar_plot(
        shap_values
    )
    
    generate_shap_report(
        model,
        X_test,
        shap_values,
        class_labels,
        health_encoder
    )

    generate_local_explanation(
        model,
        X_test,
        shap_values,
        health_encoder
    )
    
    generate_force_plot(
        model,
        X_test,
        shap_values,
        health_encoder
    )

    print("=" * 60)
    print("SHAP Analysis Completed Successfully")
    print("=" * 60)
    
def generate_summary_plots(
    shap_values,
    class_labels,
    health_encoder
):

    print("Generating SHAP Summary Plots...\n")

    for class_index, class_label in enumerate(class_labels):

        class_name = health_encoder.inverse_transform([class_label])[0]
        
        print(f"Generating {class_name} Summary Plot...")

        plt.figure(figsize=(12, 8))

        shap.plots.beeswarm(
            shap_values[:, :, class_index],
            max_display=20,
            show=False
        )

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / f"shap_summary_{class_name}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"✓ Saved : shap_summary_{class_name}.png")

    print()
    
    
def generate_global_bar_plot(shap_values):

    print("Generating Global SHAP Bar Plot...")

    # Compute mean absolute SHAP value across all samples and classes
    global_importance = (
        abs(shap_values.values)
        .mean(axis=0)      # Average over samples
        .mean(axis=1)      # Average over classes
    )

    feature_names = shap_values.feature_names

    importance_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Importance": global_importance
        })
        .sort_values("Importance", ascending=False)
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Mean |SHAP Value|")

    plt.title("Global SHAP Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "shap_bar_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✓ Saved : shap_bar_plot.png\n")
    
def generate_shap_report(
    model,
    X_train,
    shap_values,
    class_labels,
    health_encoder
):

    print("Generating SHAP Analysis Report...")

    report_path = RESULTS_DIR / "shap_report.txt"

    feature_importance = (
        abs(shap_values.values)
        .mean(axis=0)
        .mean(axis=1)
    )

    importance_df = (
        pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": feature_importance
        })
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("=" * 60 + "\n")
        file.write("SolarSentinel-X\n")
        file.write("SHAP ANALYSIS REPORT\n")
        file.write("=" * 60 + "\n\n")

        file.write("Model Information\n")
        file.write("------------------------------\n")
        file.write(f"Model Type : {type(model).__name__}\n")
        file.write(f"Training Samples : {X_train.shape[0]}\n")
        file.write(f"Number of Features : {X_train.shape[1]}\n\n")

        file.write("Detected Classes\n")
        file.write("------------------------------\n")

        for label in class_labels:

            class_name = health_encoder.inverse_transform([label])[0]

            file.write(f"{label} : {class_name}\n")

        file.write("\n")

        file.write("Top 10 Important Features\n")
        file.write("------------------------------\n")

        for _, row in importance_df.head(10).iterrows():
            file.write(
                f"{row['Feature']:<30} {row['Importance']:.6f}\n"
            )

        file.write("\n")

        file.write("Generated Files\n")
        file.write("------------------------------\n")
        file.write("shap_summary_Healthy.png\n")
        file.write("shap_summary_Warning.png\n")
        file.write("shap_summary_Critical.png\n")
        file.write("shap_bar_plot.png\n")

    print("✓ Saved : shap_report.txt\n") 
    
def generate_local_explanation(
    model,
    X_train,
    shap_values,
    health_encoder
):

    print("Generating Local SHAP Explanation...")

    sample_index = 0

    prediction = model.predict(X_train.iloc[[sample_index]])[0]

    class_name = health_encoder.inverse_transform([prediction])[0]

    explanation = shap_values[sample_index, :, prediction]

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        LOCAL_RESULTS_DIR / "waterfall_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    report_file = LOCAL_RESULTS_DIR / "prediction_report.txt"

    with open(report_file, "w", encoding="utf-8") as file:

        file.write("=" * 60 + "\n")
        file.write("LOCAL SHAP EXPLANATION\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Sample Index : {sample_index}\n")
        file.write(f"Predicted Class : {class_name}\n\n")

        file.write("Feature Values\n")
        file.write("-" * 40 + "\n")

        sample = X_train.iloc[sample_index]

        for feature, value in sample.items():
            file.write(f"{feature:<30} {value:.4f}\n")

    print("✓ Saved : waterfall_plot.png")
    print("✓ Saved : prediction_report.txt\n")
    
# ==========================================================
# Generate SHAP Force Plot
# ==========================================================

def generate_force_plot(
    model,
    X_test,
    shap_values,
    health_encoder
):

    print("Generating SHAP Force Plot...")

    sample_index = 0

    prediction = model.predict(
        X_test.iloc[[sample_index]]
    )[0]

    explanation = shap_values[
        sample_index,
        :,
        prediction
    ]

    force_plot = shap.plots.force(
        explanation,
        matplotlib=True,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        LOCAL_RESULTS_DIR / "force_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✓ Saved : force_plot.png\n")
    
    
def load_health_encoder():

    print("Loading Health Label Encoder...")

    encoder = joblib.load(ENCODER_PATH)

    print("✓ Health Label Encoder Loaded\n")

    return encoder
    

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "real_training",
    "real_sensor_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "random_forest_real.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "real_scaler.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "feature_names_real.csv"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest",
    "real_trained",
    "health_encoder_real.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "real_shap"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("SOLARSENTINEL-X")
print("REAL SENSOR SHAP EXPLAINABILITY")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully!")
print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

feature_names = pd.read_csv(
    FEATURE_PATH
).iloc[:, 0].dropna().tolist()

print("\nReal sensor features:")

for feature in feature_names:
    print(" -", feature)

print("\nNumber of features :", len(feature_names))


# ============================================================
# PREPARE DATA
# ============================================================

X = df[feature_names].apply(
    pd.to_numeric,
    errors="coerce"
)

valid_mask = X.notna().all(axis=1)

X = X.loc[valid_mask].copy()
df_valid = df.loc[valid_mask].copy()

print("\nValid samples :", len(X))
print("Feature shape :", X.shape)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Random Forest loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

X_scaled = scaler.transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_names,
    index=X.index
)

print("Scaling completed.")


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

print("\nLoading health encoder...")

encoder = joblib.load(ENCODER_PATH)

print("Health classes:")

for index, label in enumerate(encoder.classes_):
    print(index, "-->", label)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

predictions = model.predict(X_scaled)

prediction_labels = encoder.inverse_transform(
    predictions.astype(int)
)

df_valid["Predicted_Health"] = prediction_labels


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("SHAP explainer created successfully.")


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(
    X_scaled_df
)

print("SHAP calculation completed.")


# ============================================================
# HANDLE SHAP OUTPUT FORMAT
# ============================================================

if isinstance(shap_values, list):

    print("\nSHAP format detected: list")

    shap_values_array = np.array(shap_values)

else:

    print("\nSHAP format detected: ndarray")

    shap_values_array = np.asarray(shap_values)


print(
    "SHAP array shape:",
    shap_values_array.shape
)


# ============================================================
# SAVE RAW SHAP VALUES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "shap_values_real.npy"
    ),
    shap_values_array
)

print("\nSaved:")
print(
    os.path.join(
        OUTPUT_DIR,
        "shap_values_real.npy"
    )
)


# ============================================================
# FEATURE IMPORTANCE FROM RANDOM FOREST
# ============================================================

rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Random_Forest_Importance": model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "Random_Forest_Importance",
    ascending=False
)

rf_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "random_forest_feature_importance.csv"
    ),
    index=False
)

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

print(rf_importance.to_string(index=False))


# ============================================================
# DETERMINE SHAP CLASS STRUCTURE
# ============================================================

number_of_classes = len(
    encoder.classes_
)

print("\nNumber of classes :", number_of_classes)


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

if shap_values_array.ndim == 3:

    # Possible formats:
    # (samples, features, classes)
    # OR
    # (classes, samples, features)

    if shap_values_array.shape[0] == len(X_scaled_df):

        # samples, features, classes
        global_mean_abs = np.mean(
            np.abs(shap_values_array),
            axis=(0, 2)
        )

    else:

        # classes, samples, features
        global_mean_abs = np.mean(
            np.abs(shap_values_array),
            axis=(0, 1)

        )

else:

    global_mean_abs = np.mean(
        np.abs(shap_values_array),
        axis=0
    )


global_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": global_mean_abs
})

global_importance = global_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

global_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "global_shap_feature_importance.csv"
    ),
    index=False
)


print("\n" + "=" * 70)
print("GLOBAL SHAP FEATURE IMPORTANCE")
print("=" * 70)

print(
    global_importance.to_string(index=False)
)


# ============================================================
# CLASS-SPECIFIC SHAP VALUES
# ============================================================

for class_index, class_name in enumerate(
    encoder.classes_
):

    print("\n" + "=" * 70)
    print(
        "SHAP ANALYSIS:",
        class_name
    )
    print("=" * 70)

    # Extract SHAP values for current class
    if shap_values_array.ndim == 3:

        if shap_values_array.shape[0] == len(X_scaled_df):

            class_shap = shap_values_array[
                :, :,
                class_index
            ]

        else:

            class_shap = shap_values_array[
                class_index,
                :, :
            ]

    else:

        class_shap = shap_values_array


    # --------------------------------------------------------
    # CLASS FEATURE IMPORTANCE
    # --------------------------------------------------------

    class_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": np.mean(
            np.abs(class_shap),
            axis=0
        )
    })

    class_importance = class_importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    safe_class_name = str(
        class_name
    ).replace(" ", "_")

    class_csv_path = os.path.join(
        OUTPUT_DIR,
        f"shap_importance_{safe_class_name}.csv"
    )

    class_importance.to_csv(
        class_csv_path,
        index=False
    )

    print(
        "\nFeature importance for",
        class_name
    )

    print(
        class_importance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SHAP SUMMARY BAR
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    shap.summary_plot(
        class_shap,
        X_scaled_df,
        plot_type="bar",
        show=False
    )

    plt.title(
        f"SHAP Feature Importance - {class_name}"
    )

    plt.tight_layout()

    bar_path = os.path.join(
        OUTPUT_DIR,
        f"shap_bar_{safe_class_name}.png"
    )

    plt.savefig(
        bar_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        bar_path
    )

    # --------------------------------------------------------
    # SHAP SUMMARY DOT PLOT
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        class_shap,
        X_scaled_df,
        show=False
    )

    plt.title(
        f"SHAP Summary - {class_name}"
    )

    plt.tight_layout()

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"shap_summary_{safe_class_name}.png"
    )

    plt.savefig(
        summary_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        summary_path
    )


# ============================================================
# TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP REAL SENSOR FEATURES")
print("=" * 70)

top_features = global_importance.head(10)

for _, row in top_features.iterrows():

    print(
        f"{row['Feature']:<30} "
        f"{row['Mean_Absolute_SHAP']:.6f}"
    )


# ============================================================
# SAVE PREDICTION + SHAP SUMMARY
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "real_sensor_predictions.csv"
)

df_valid[
    feature_names
    + ["System_Health", "Predicted_Health"]
].to_csv(
    prediction_path,
    index=False
)

print("\nSaved:")
print(prediction_path)


# ============================================================
# TEXT REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "real_shap_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "SolarSentinel-X Real Sensor SHAP Report\n"
    )

    report.write(
        "=" * 60 + "\n\n"
    )

    report.write(
        f"Total valid samples: {len(X)}\n"
    )

    report.write(
        f"Number of features: {len(feature_names)}\n\n"
    )

    report.write(
        "Features:\n"
    )

    for feature in feature_names:
        report.write(
            f"- {feature}\n"
        )

    report.write(
        "\nGlobal SHAP Feature Importance:\n"
    )

    report.write(
        global_importance.to_string(
            index=False
        )
    )

    report.write(
        "\n\nRandom Forest Feature Importance:\n"
    )

    report.write(
        rf_importance.to_string(
            index=False
        )
    )

    report.write(
        "\n\nClass Labels:\n"
    )

    for index, label in enumerate(
        encoder.classes_
    ):
        report.write(
            f"{index} -> {label}\n"
        )


print("\nSaved:")
print(report_path)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("REAL SENSOR SHAP ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated output directory:")
print(OUTPUT_DIR)

print("\nImportant files:")

print("1. shap_values_real.npy")
print("2. global_shap_feature_importance.csv")
print("3. random_forest_feature_importance.csv")
print("4. shap_importance_Critical.csv")
print("5. shap_importance_Healthy.csv")
print("6. shap_importance_Warning.csv")
print("7. shap_bar_Critical.png")
print("8. shap_bar_Healthy.png")
print("9. shap_bar_Warning.png")
print("10. shap_summary_Critical.png")
print("11. shap_summary_Healthy.png")
print("12. shap_summary_Warning.png")
print("13. real_sensor_predictions.csv")
print("14. real_shap_report.txt")

print("\nPhase 14 SHAP processing finished.")
# -- coding: utf-8 --
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CLASS_CONFIG,
    CONFUSION_MATRIX_PLOT_PATH,
    FEATURE_METADATA_PATH,
    MODEL_NAME,
    PLOT_OUTPUT_DIR,
    ROC_PLOT_PATH,
)


def get_class_names():
    positive_name = next(
        (name for name, label_value in CLASS_CONFIG.items() if label_value == 1),
        "A",
    )
    negative_name = next(
        (name for name, label_value in CLASS_CONFIG.items() if label_value == 0),
        "B",
    )
    return negative_name, positive_name


def configure_plot_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def draw_roc_curve(fpr, tpr, roc_auc, title, output_path):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot([0, 1], [0, 1], "--", color=(0.7, 0.7, 0.7))
    ax.plot(
        fpr,
        tpr,
        "k--",
        label="ROC (area = %0.2f)" % roc_auc,
        lw=2,
    )
    ax.set_xlim([0.00, 1.00])
    ax.set_ylim([0.00, 1.00])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=18)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_PLOT_PATH if output_path is None else output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def load_feature_selection_outputs():
    with open(FEATURE_METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    train_data = pd.read_csv(metadata["train_selected_path"])
    test_data = pd.read_csv(metadata["test_selected_path"])
    selected_features = metadata["selected_features"]
    return metadata, train_data, test_data, selected_features


def build_classifier(model_name, random_state):
    if model_name == "forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=30, random_state=random_state)
    if model_name == "svm":
        from sklearn import svm

        return svm.SVC(kernel="rbf", gamma="auto", probability=True)
    if model_name == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier

        return AdaBoostClassifier(n_estimators=100, algorithm="SAMME.R")
    if model_name == "decisiontree":
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(criterion="gini")
    if model_name == "bayes":
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB()
    if model_name == "MLP":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            solver="lbfgs",
            alpha=1e-5,
            hidden_layer_sizes=(15, 2),
            random_state=1,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def train_classifier(train_data, selected_features, model_name, random_state):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    x = train_data[selected_features]
    y = train_data["label"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=selected_features, index=x.index)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_scaled,
        y,
        test_size=0.1,
        random_state=random_state,
        stratify=y,
    )

    model = build_classifier(model_name, random_state)
    model.fit(x_train, y_train)

    score = model.score(x_valid, y_valid)
    print("Model:", model_name)
    print("Validation accuracy: {}".format(score))
    return model, scaler, score


def save_model_bundle(model, scaler, model_name, data_path, selected_features):
    import joblib

    bundle = {
        "model": model,
        "scaler": scaler,
        "selected_features": list(selected_features),
    }
    model_path = os.path.join(data_path, "model_" + model_name + ".model")
    joblib.dump(bundle, model_path)
    print("Saved model bundle to:", model_path)
    return model_path


def evaluate_model(model, scaler, test_data, selected_features):
    from sklearn.metrics import (
        auc,
        classification_report,
        confusion_matrix,
        roc_curve,
    )
    import seaborn as sns

    x_test_data = test_data[selected_features].astype(np.float32)
    x_test_data = scaler.transform(x_test_data)
    x_test_data = pd.DataFrame(x_test_data, columns=selected_features)
    y_test_data = test_data["label"]
    negative_name, positive_name = get_class_names()

    print(
        "Test samples: {}, features: {}".format(
            len(x_test_data),
            x_test_data.shape[1],
        )
    )

    score = model.score(x_test_data, y_test_data)
    print("Test accuracy: {}".format(score))

    predict_label = model.predict(x_test_data)
    label = y_test_data.to_list()
    confusion = confusion_matrix(label, predict_label)
    row_totals = confusion.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    confusion_ratio = confusion / row_totals
    confusion_text = np.array(
        [
            [
                "{}".format(confusion[row, col])
                for col in range(confusion.shape[1])
            ]
            for row in range(confusion.shape[0])
        ]
    )

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    sns.heatmap(
        confusion,
        cmap=sns.light_palette("#0f4c81", as_cmap=True),
        annot=False,
        cbar=True,
        square=True,
        linewidths=1.5,
        linecolor="white",
        xticklabels=[negative_name, positive_name],
        yticklabels=[negative_name, positive_name],
        ax=ax,
    )
    max_value = confusion.max() if confusion.size else 0
    threshold = max_value * 0.45
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            text_color = "white" if confusion[row, col] > threshold else "#1f2937"
            ax.text(
                col + 0.5,
                row + 0.5,
                confusion_text[row, col],
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color=text_color,
            )
    ax.set_title("Confusion Matrix", pad=12, weight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved confusion matrix plot to:", CONFUSION_MATRIX_PLOT_PATH)

    kind = dict(CLASS_CONFIG)
    if hasattr(model, "predict_proba"):
        y_predict = model.predict_proba(x_test_data)
        fpr_positive, tpr_positive, _ = roc_curve(
            label,
            y_predict[:, kind[positive_name]],
            pos_label=kind[positive_name],
        )
        auc_positive = auc(fpr_positive, tpr_positive)
    elif hasattr(model, "decision_function"):
        positive_scores = model.decision_function(x_test_data)
        fpr_positive, tpr_positive, _ = roc_curve(
            label,
            positive_scores,
            pos_label=kind[positive_name],
        )
        auc_positive = auc(fpr_positive, tpr_positive)
    else:
        raise ValueError("The current model does not support ROC curve plotting.")

    draw_roc_curve(
        fpr_positive,
        tpr_positive,
        auc_positive,
        title="ROC Curve ({})".format(positive_name),
        output_path=ROC_PLOT_PATH,
    )
    print("Saved ROC curve plot to:", ROC_PLOT_PATH)

    print("Confusion matrix:\n{}".format(confusion))
    print("\nClassification metrics:")
    print(classification_report(label, predict_label))
    return {
        "score": score,
        "roc_auc_positive": auc_positive,
        "confusion": confusion,
        "labels": label,
        "predictions": predict_label,
    }


def main():
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    metadata, train_data, test_data, selected_features = load_feature_selection_outputs()
    if not selected_features:
        raise ValueError("selected_features.json does not contain any selected features.")

    model, scaler, _ = train_classifier(
        train_data,
        selected_features,
        MODEL_NAME,
        metadata["random_state"],
    )
    save_model_bundle(
        model,
        scaler,
        MODEL_NAME,
        metadata["data_path"],
        selected_features,
    )
    evaluate_model(model, scaler, test_data, selected_features)


if __name__ == "__main__":
    main()

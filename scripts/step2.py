# -- coding: utf-8 --
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from config import (
    CLASS_CONFIG,
    DATA_PATH,
    FEATURE_METADATA_PATH,
    MAX_MUSE_FEATURES,
    RANDOM_STATE,
    SCRIPT_DIR,
    TEST_RATIO,
    TEST_SELECTED_PATH,
    TRAIN_SELECTED_PATH,
    USE_MUSE,
    USE_TTEST,
)


def split_df(df, ratio):
    cut_idx = int(round(ratio * df.shape[0]))
    print(cut_idx)
    data_test, data_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
    return data_train, data_test


def load_group_data(data_path, group_name, label, random_state):
    file_path = os.path.join(
        data_path,
        group_name,
        "results",
        "extracted_radiomics_features.xlsx",
    )
    data = pd.read_excel(file_path)
    data.insert(0, "label", label)
    return data.sample(frac=1.0, random_state=random_state)


def find_dim_columns(columns):
    dim_columns = [column for column in columns if "dim" in column]
    print("cols_to_remove:", dim_columns)
    return dim_columns


def drop_string_columns(df):
    string_columns = []
    for column in df.columns:
        series = df[column].dropna()
        if not series.empty and isinstance(series.iloc[0], str):
            string_columns.append(column)
    return df.drop(columns=string_columns), string_columns


def clean_group_data(df, dim_columns):
    cleaned = df.drop(columns=dim_columns, errors="ignore")
    cleaned, string_columns = drop_string_columns(cleaned)
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.dropna(axis=1, how="any")
    return cleaned, string_columns


def prepare_group_datasets(data_path, class_config, random_state):
    group_frames = {}
    raw_frames = {}
    for group_name, label in class_config.items():
        raw_frames[group_name] = load_group_data(data_path, group_name, label, random_state)

    dim_columns = find_dim_columns(raw_frames["A"].columns)
    for group_name, raw_df in raw_frames.items():
        cleaned_df, string_columns = clean_group_data(raw_df, dim_columns)
        print(f"{group_name} string columns removed:", string_columns)
        print(f"{group_name}_data shape:", cleaned_df.shape)
        group_frames[group_name] = cleaned_df
    return group_frames


def split_group_datasets(group_frames, test_ratio):
    train_frames = {}
    test_frames = {}
    for group_name, df in group_frames.items():
        train_frames[group_name], test_frames[group_name] = split_df(df, test_ratio)
    return train_frames, test_frames


def save_group_test_sets(test_frames):
    for group_name, df in test_frames.items():
        output_path = os.path.join(SCRIPT_DIR, f"{group_name}_test.csv")
        df.to_csv(output_path, index=False)


def combine_and_shuffle_frames(frames, random_state):
    data = pd.concat(frames, axis=0)
    return data.sample(frac=1.0, random_state=random_state)


def run_ttest_feature_selection(a_train, b_train, use_ttest=True):
    columns_index = []
    print("Start T-test feature selection")
    if not use_ttest:
        return list(a_train.columns[1:])

    from scipy.stats import levene, ttest_ind

    for column_name in a_train.columns[1:]:
        a_values = a_train[column_name]
        b_values = b_train[column_name]
        same_variance = levene(a_values, b_values)[1] > 0.05
        p_value = ttest_ind(a_values, b_values, equal_var=same_variance)[1]
        if p_value < 0.05:
            columns_index.append(column_name)

    print("columns_index:", columns_index)
    print("Features left after T-test:", len(columns_index))
    return columns_index


def run_muse_selection(data, columns_index, max_columns_num, use_muse=False):
    if not use_muse:
        return columns_index

    try:
        from kydavra import MUSESelector
    except ModuleNotFoundError:
        print("kydavra is not installed, skip MUSE feature selection.")
        return columns_index

    print("Start MUSE feature selection")
    selector = MUSESelector(num_features=max_columns_num)
    selected_columns = selector.select(data, "label")
    if "label" in selected_columns:
        return selected_columns[1:]
    return selected_columns


def build_feature_subset(a_train, b_train, feature_columns, random_state):
    selected_columns = list(feature_columns)
    if "label" not in selected_columns:
        selected_columns = ["label"] + selected_columns

    a_subset = a_train[selected_columns]
    b_subset = b_train[selected_columns]
    data = combine_and_shuffle_frames([a_subset, b_subset], random_state)
    return data, selected_columns


def run_lasso_feature_selection(data):
    x = data[data.columns[1:]]
    y = data["label"]
    column_names = x.columns

    lasso_x = x.astype(np.float32)
    scaler = StandardScaler()
    lasso_x = scaler.fit_transform(lasso_x)
    lasso_x = pd.DataFrame(lasso_x, columns=column_names)

    alpha_range = np.logspace(-3, 1, 50)
    model = LassoCV(alphas=alpha_range, cv=5, max_iter=100000)
    model.fit(lasso_x, y)

    print(model.alpha_)
    coef = pd.Series(model.coef_, index=column_names)
    selected_features = list(coef[coef != 0].index)
    print(
        "Selected {} features from {}".format(
            len(selected_features), len(column_names)
        )
    )
    print("Selected features:")
    print(coef[coef != 0])
    return {
        "x": lasso_x,
        "y": y,
        "coef": coef,
        "model": model,
        "alpha_range": alpha_range,
        "scaler": scaler,
        "selected_features": selected_features,
    }


def plot_lasso_diagnostics(lasso_result, data):
    selected_features = lasso_result["selected_features"]
    coef = lasso_result["coef"]
    lasso_model = lasso_result["model"]
    alpha_range = lasso_result["alpha_range"]
    scaler = lasso_result["scaler"]

    if not selected_features:
        print("Skip plots because no features were selected by Lasso.")
        return

    selected_x = lasso_result["x"][selected_features]

    import seaborn as sns

    _, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        selected_x.corr(),
        annot=True,
        cmap="coolwarm",
        annot_kws={"size": 10, "weight": "bold"},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, va="top", ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)

    weight = coef[coef != 0].to_dict()
    weight = dict(sorted(weight.items(), key=lambda item: item[1], reverse=False))
    plt.figure(figsize=(8, 6))
    plt.title("characters classification weight", fontsize=15)
    plt.xlabel("weighted value", fontsize=14)
    plt.ylabel("feature")
    plt.barh(
        range(len(weight.values())),
        list(weight.values()),
        tick_label=list(weight.keys()),
        alpha=0.6,
        facecolor="blue",
        edgecolor="black",
        label="feature weight",
    )
    plt.legend(loc=4)

    mses = lasso_model.mse_path_
    mse = [np.mean(m) for m in mses]
    std = [np.std(m) for m in mses]

    plt.figure(figsize=(8, 6))
    plt.errorbar(
        lasso_model.alphas_,
        mse,
        std,
        fmt="o:",
        ecolor="lightblue",
        elinewidth=3,
        ms=5,
        mfc="wheat",
        mec="salmon",
        capsize=3,
    )
    plt.axvline(lasso_model.alpha_, color="red", ls="--")
    plt.title("Errorbar")
    plt.xlabel("Lambda")
    plt.ylabel("MSE")

    x = data[data.columns[1:]].astype(np.float32)
    y = data["label"]
    x = scaler.transform(x)
    x = pd.DataFrame(x, columns=data.columns[1:])
    coefs = lasso_model.path(x, y, alphas=alpha_range, max_iter=1000)[1].T

    plt.figure(figsize=(8, 6))
    plt.plot(lasso_model.alphas_, coefs, "-")
    plt.axvline(lasso_model.alpha_, color="red", ls="--")
    plt.xlabel("Lambda")
    plt.ylabel("coef")


def save_selected_outputs(train_frames, test_frames, selected_features, random_state):
    selected_columns = ["label"] + list(selected_features)
    train_selected = combine_and_shuffle_frames(
        [train_frames["A"][selected_columns], train_frames["B"][selected_columns]],
        random_state,
    )
    test_selected = combine_and_shuffle_frames(
        [test_frames["A"][selected_columns], test_frames["B"][selected_columns]],
        random_state,
    )

    train_selected.to_csv(TRAIN_SELECTED_PATH, index=False)
    test_selected.to_csv(TEST_SELECTED_PATH, index=False)

    metadata = {
        "data_path": DATA_PATH,
        "class_config": CLASS_CONFIG,
        "test_ratio": TEST_RATIO,
        "random_state": RANDOM_STATE,
        "use_ttest": USE_TTEST,
        "use_muse": USE_MUSE,
        "max_muse_features": MAX_MUSE_FEATURES,
        "selected_features": list(selected_features),
        "train_selected_path": TRAIN_SELECTED_PATH,
        "test_selected_path": TEST_SELECTED_PATH,
    }
    with open(FEATURE_METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("Saved selected training data to:", TRAIN_SELECTED_PATH)
    print("Saved selected test data to:", TEST_SELECTED_PATH)
    print("Saved feature metadata to:", FEATURE_METADATA_PATH)


def main():
    group_frames = prepare_group_datasets(DATA_PATH, CLASS_CONFIG, RANDOM_STATE)
    train_frames, test_frames = split_group_datasets(group_frames, TEST_RATIO)
    save_group_test_sets(test_frames)

    train_data = combine_and_shuffle_frames(
        [train_frames["A"], train_frames["B"]],
        RANDOM_STATE,
    )
    print("Training rows: {}".format(len(train_data)))
    print("Training columns: {}".format(train_data.shape[1]))

    feature_columns = run_ttest_feature_selection(
        train_frames["A"],
        train_frames["B"],
        use_ttest=USE_TTEST,
    )
    if not feature_columns:
        raise ValueError("No features were selected after T-test filtering.")

    subset_data, _ = build_feature_subset(
        train_frames["A"],
        train_frames["B"],
        feature_columns,
        RANDOM_STATE,
    )

    feature_columns = run_muse_selection(
        subset_data,
        feature_columns,
        MAX_MUSE_FEATURES,
        use_muse=USE_MUSE,
    )
    print("Features left after MUSE:", len(feature_columns))
    if not feature_columns:
        raise ValueError("No features were selected after MUSE filtering.")

    subset_data, _ = build_feature_subset(
        train_frames["A"],
        train_frames["B"],
        feature_columns,
        RANDOM_STATE,
    )
    lasso_result = run_lasso_feature_selection(subset_data)
    plot_lasso_diagnostics(lasso_result, subset_data)

    selected_features = lasso_result["selected_features"]
    if not selected_features:
        raise ValueError("No features were selected by Lasso.")

    save_selected_outputs(
        train_frames,
        test_frames,
        selected_features,
        RANDOM_STATE,
    )


if __name__ == "__main__":
    main()

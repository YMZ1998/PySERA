import copy
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_CONFIG_PATH = os.path.join(SCRIPT_DIR, "runtime_config.json")
MODEL_OPTIONS = ["bayes", "svm", "forest", "adaboost", "decisiontree", "MLP"]

DEFAULT_CONFIG = {
    "root_dir": r"E:\AI-data2\Radiomics",
    "dataset_name": r"Dataset0978_test",
    "class_config": {"A": 1, "B": 0},
    "test_ratio": 0.15,
    "random_state": 114,
    "use_ttest": True,
    "use_muse": True,
    "max_muse_features": 60,
    "model_name": "MLP",
    "pysera_process_kwargs": {
        "num_workers": "8",
        "enable_parallelism": True,
        "categories": "all",
        "bin_size": 25,
        "roi_num": 1,
        "roi_selection_mode": "per_region",
        "min_roi_volume": 5,
        "apply_preprocessing": True,
        "feature_value_mode": "APPROXIMATE_VALUE",
        "IBSI_based_parameters": {
            "radiomics_DataType": "CT",
            "radiomics_DiscType": "FBN",
            "radiomics_isScale": 1,
        },
        "report": "info",
    },
}


def _deep_merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_runtime_overrides():
    if not os.path.exists(RUNTIME_CONFIG_PATH):
        return {}
    with open(RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_runtime_config():
    return _deep_merge(DEFAULT_CONFIG, load_runtime_overrides())


def save_runtime_overrides(overrides):
    with open(RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(overrides, file, ensure_ascii=False, indent=2)


def update_runtime_config(updates):
    overrides = load_runtime_overrides()
    merged = _deep_merge(overrides, updates)
    save_runtime_overrides(merged)
    return get_runtime_config()


_CONFIG = get_runtime_config()

ROOT_DIR = _CONFIG["root_dir"]
DATA_PATH = os.path.join(ROOT_DIR, _CONFIG["dataset_name"])
CLASS_CONFIG = _CONFIG["class_config"]
TEST_RATIO = _CONFIG["test_ratio"]
RANDOM_STATE = _CONFIG["random_state"]
USE_TTEST = _CONFIG["use_ttest"]
USE_MUSE = _CONFIG["use_muse"]
MAX_MUSE_FEATURES = _CONFIG["max_muse_features"]
MODEL_NAME = _CONFIG["model_name"]
PYSERA_PROCESS_KWARGS = _CONFIG["pysera_process_kwargs"]

TRAIN_SELECTED_PATH = os.path.join(DATA_PATH, "train_selected.csv")
TEST_SELECTED_PATH = os.path.join(DATA_PATH, "test_selected.csv")
FEATURE_METADATA_PATH = os.path.join(DATA_PATH, "selected_features.json")
PLOT_OUTPUT_DIR = os.path.join(DATA_PATH, "plots")
LOG_DIR = os.path.join(DATA_PATH, "logs")
WEBAPP_STDOUT_LOG_PATH = os.path.join(LOG_DIR, "webapp.out.log")
WEBAPP_STDERR_LOG_PATH = os.path.join(LOG_DIR, "webapp.err.log")
ROC_PLOT_PATH = os.path.join(PLOT_OUTPUT_DIR, "step3_roc_curve.png")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(PLOT_OUTPUT_DIR, "step3_confusion_matrix.png")

if __name__ == "__main__":
    print(_CONFIG)
    print(DATA_PATH)

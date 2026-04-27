import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = r"E:\AI-data2\nnUNet_raw\Dataset0978_test"
CLASS_CONFIG = {"A": 1, "B": 0}

TEST_RATIO = 0.15
RANDOM_STATE = 114

USE_TTEST = True
USE_MUSE = True
MAX_MUSE_FEATURES = 60

MODEL_NAME = "bayes"

PYSERA_PROCESS_KWARGS = {
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
}

TRAIN_SELECTED_PATH = os.path.join(SCRIPT_DIR, "train_selected.csv")
TEST_SELECTED_PATH = os.path.join(SCRIPT_DIR, "test_selected.csv")
FEATURE_METADATA_PATH = os.path.join(SCRIPT_DIR, "selected_features.json")
PLOT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "plots")
ROC_PLOT_PATH = os.path.join(PLOT_OUTPUT_DIR, "step3_roc_curve.png")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(PLOT_OUTPUT_DIR, "step3_confusion_matrix.png")

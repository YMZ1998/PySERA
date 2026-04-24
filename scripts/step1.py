import os.path

import pysera

data_path = r"E:\AI-data2\nnUNet_raw\Dataset0978_test"

class_dic = ["A", "B"]

for i in class_dic:
    # Comprehensive processing with custom parameters
    result = pysera.process_batch(
        image_input=os.path.join(data_path, i, "imagesTr"),
        mask_input=os.path.join(data_path, i, "labelsTr"),
        output_path=os.path.join(data_path, i, "results"),

        # Performance settings
        num_workers="8",  # Use 2 CPU cores
        enable_parallelism=True,  # Disable multiprocessing

        # Image feature extraction settings
        # categories="glcm, glrlm, glszm",  # Extract specific texture feature categories
        # dimensions="1st, 2_5d, 3d",  # Extract features in 1st order, 2.5D and 3D dimensions
        # Alternative examples for categories and dimensions:
        categories="all",                 # Extract all 557 features
        # categories="stat, morph, glcm",   # Statistical, morphological and GLCM features
        # dimensions="2D",                  # Extract only 2D features
        # dimensions="all",                 # Extract features in all dimensions

        bin_size=25,  # Texture analysis bin size
        roi_num=1,  # Number of ROIs to process
        roi_selection_mode="per_region",  # ROI selection strategy
        min_roi_volume=5,  # Minimum ROI volume threshold

        # Processing options
        apply_preprocessing=True,  # Apply ROI preprocessing
        feature_value_mode="APPROXIMATE_VALUE",  # Strategy for handling NaN values.

        # IBSI parameters (advanced, overrides defaults)
        IBSI_based_parameters={
            "radiomics_DataType": "CT",
            "radiomics_DiscType": "FBN",
            "radiomics_isScale": 1
        },

        # Logging options
        report="info"  # Report detail level: "all" (full processing details),
        # "info" (essential information), "warning" (warnings only),
        # "error" (errors only), "none" (no reporting). Default: "all"
    )

    print(f"Success: {result['success']}")
    print(f"Features extracted: {result['features_extracted']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")

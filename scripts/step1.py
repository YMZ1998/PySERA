import os.path

import pysera

from config import CLASS_CONFIG, DATA_PATH, PYSERA_PROCESS_KWARGS


for group_name in CLASS_CONFIG:
    # Comprehensive processing with custom parameters
    result = pysera.process_batch(
        image_input=os.path.join(DATA_PATH, group_name, "imagesTr"),
        mask_input=os.path.join(DATA_PATH, group_name, "labelsTr"),
        output_path=os.path.join(DATA_PATH, group_name, "results"),
        **PYSERA_PROCESS_KWARGS,
    )

    print(f"Success: {result['success']}")
    print(f"Features extracted: {result['features_extracted']}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")

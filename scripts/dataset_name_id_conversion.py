from os.path import isdir

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

from scripts.config import ROOT_DIR


def find_candidate_datasets(dataset_id: int):
    startswith = "Dataset%04.0d" % dataset_id

    if ROOT_DIR is not None and isdir(ROOT_DIR):
        candidates = subdirs(ROOT_DIR, prefix=startswith, join=False)
    else:
        candidates = []
    candidates = np.unique(candidates)
    return candidates


def convert_id_to_dataset_name(dataset_id: int):
    unique_candidates = find_candidate_datasets(dataset_id)
    if len(unique_candidates) > 1:
        raise RuntimeError("More than one dataset name found for dataset id %d. " % (dataset_id))
    if len(unique_candidates) == 0:
        raise RuntimeError(f"Could not find a dataset with the ID {dataset_id}")
    return unique_candidates[0]


def convert_dataset_name_to_id(dataset_name: str):
    assert dataset_name.startswith("Dataset")
    dataset_id = int(dataset_name[7:11])
    return dataset_id


def maybe_convert_to_dataset_name(dataset_name_or_id: Union[int, str]) -> str:
    if isinstance(dataset_name_or_id, str) and dataset_name_or_id.startswith("Dataset"):
        return dataset_name_or_id
    if isinstance(dataset_name_or_id, str):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id was a string and did not start with 'Dataset' so we tried to "
                             "convert it to a dataset ID (int). That failed, however. Please give an integer number "
                             "('1', '2', etc) or a correct dataset name. Your input: %s" % dataset_name_or_id)
    return str(convert_id_to_dataset_name(int(dataset_name_or_id)))


def find_all_datasets():
    all_ids = [convert_dataset_name_to_id(dataset_name) for dataset_name in subdirs(ROOT_DIR, join=False)]
    print("find all datasets ids: ", all_ids)
    return all_ids


if __name__ == "__main__":
    print(convert_id_to_dataset_name(978))
    print(convert_dataset_name_to_id("Dataset0978_test"))
    print(find_all_datasets())

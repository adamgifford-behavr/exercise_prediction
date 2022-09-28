import json
import os
from pathlib import Path

import pandas as pd
import pytest
from deepdiff import DeepDiff
from dotenv import find_dotenv, load_dotenv

import src.features.build_features as bf

load_dotenv(find_dotenv())
PROJDIR = Path(__file__).resolve().parents[1]
TABLE_COLUMNS = [
    "featurize_id",
    "file",
    "dataset_group",
    "added_datetime",
    "window_size",
    "t_index",
    "label",
    "label_group",
    "accel_x_0",
    "accel_x_1",
    "accel_x_2",
    "accel_x_4",
    "accel_x_5",
    "accel_x_7",
    "accel_x_9",
    "accel_x_12",
    "accel_x_16",
    "accel_y_0",
    "accel_y_1",
    "accel_y_2",
    "accel_y_3",
    "accel_y_4",
    "accel_y_5",
    "accel_y_6",
    "accel_y_7",
    "accel_y_8",
    "accel_y_9",
    "accel_y_10",
    "accel_y_11",
    "accel_y_12",
    "accel_z_0",
    "accel_z_1",
    "accel_z_2",
    "accel_z_3",
    "accel_z_4",
    "accel_z_5",
    "accel_z_8",
    "accel_z_12",
    "accel_z_14",
    "gyro_x_0",
    "gyro_x_1",
    "gyro_x_2",
    "gyro_x_3",
    "gyro_x_4",
    "gyro_x_5",
    "gyro_x_6",
    "gyro_x_7",
    "gyro_x_8",
    "gyro_x_9",
    "gyro_x_10",
    "gyro_x_11",
    "gyro_x_12",
    "gyro_x_13",
    "gyro_y_0",
    "gyro_y_1",
    "gyro_y_2",
    "gyro_y_3",
    "gyro_y_4",
    "gyro_y_5",
    "gyro_y_6",
    "gyro_y_7",
    "gyro_y_8",
    "gyro_y_10",
    "gyro_y_11",
    "gyro_z_0",
    "gyro_z_1",
    "gyro_z_3",
    "gyro_z_4",
    "gyro_z_5",
    "gyro_z_8",
    "gyro_z_14",
    "gyro_z_17",
    "gyro_z_18",
]


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data


## src/features/build_features.py
# featurize_id
@pytest.mark.xfail(
    reason="will not run on cloned repo w/o raw data, make data commands"
)
def test_generate_featurize_id():
    datafile_splits = read_json(PROJDIR / "src/features/datafile_group_splits.json")
    metaparams = read_json(PROJDIR / "src/features/metaparams.json")

    actual_result = bf._generate_featurize_id(datafile_splits, metaparams)
    expected_result = os.getenv("FEATURIZE_ID")

    assert (
        actual_result == expected_result
    ), "featurize_id for same input parameters differs"

    metaparams["n_fft"] = 152
    different_result = bf._generate_featurize_id(datafile_splits, metaparams)

    assert (
        different_result != expected_result
    ), "featurize_id for different input parameters is equivalent"


# calculate windowed feats
# there are deprecation warnings internal to fastparquet for loading the data, ignore
# for testing as these are not relevant
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.xfail(
    reason="will not run on cloned repo w/o raw data, make data commands"
)
def test_calculate_windowed_feats():
    featurize_id = os.getenv("FEATURIZE_ID")
    dataset_group = "train"
    df_file = PROJDIR / "data/interim/fileID1_subjID3_dataID0.parquet"
    features = read_json(PROJDIR / "src/features/frequency_features.json")
    n_fft = 151

    actual_df = bf.calculate_windowed_feats(
        featurize_id, dataset_group, df_file, n_fft, features, TABLE_COLUMNS
    )
    # this is done because added_datetime changes whenever the code is run
    actual_df = actual_df.drop(columns=["added_datetime"])
    actual_shape = actual_df.shape
    actual_head = actual_df.head().to_dict("records")
    actual_tail = actual_df.tail().to_dict("records")

    expected_records = read_json(PROJDIR / "tests/expected_features_df_data.json")
    expected_head = expected_records["head"]
    expected_tail = expected_records["tail"]

    # ignoring files here because the absolute paths may vary by system
    head_diff = DeepDiff(
        expected_head,
        actual_head,
        significant_digits=3,
        exclude_regex_paths={
            r"root\[\d+\]\['file'\]",
            r"root\[\d+\]\['featurize_id'\]",
        },
    )
    tail_diff = DeepDiff(
        expected_tail,
        actual_tail,
        significant_digits=3,
        exclude_regex_paths={
            r"root\[\d+\]\['file'\]",
            r"root\[\d+\]\['featurize_id'\]",
        },
    )

    assert actual_shape == (847, 72), "size different"

    assert "values_changed" not in head_diff, "values_changed"
    assert "iterable_item_added" not in head_diff, "iterable_item_added"
    assert "iterable_item_removed" not in head_diff, "iterable_item_removed"
    assert "dictionary_item_added" not in head_diff, "dictionary_item_added"
    assert "dictionary_item_removed" not in head_diff, "dictionary_item_removed"

    assert "values_changed" not in tail_diff, "values_changed"
    assert "iterable_item_added" not in tail_diff, "iterable_item_added"
    assert "iterable_item_removed" not in tail_diff, "iterable_item_removed"
    assert "dictionary_item_added" not in tail_diff, "dictionary_item_added"
    assert "dictionary_item_removed" not in tail_diff, "dictionary_item_removed"

import json
from pathlib import Path

import pandas as pd
from deepdiff import DeepDiff

import src.models.score_batch as sb

PROJDIR = Path(__file__).resolve().parents[1]


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data


# test process columns
def test_process_columns():
    non_feature_cols = [
        "naive_frequency_features_id",
        "featurize_id",
        "file",
        "dataset_group",
        "added_datetime",
        "window_size",
        "t_index",
        "label",
        "label_group",
    ]
    expected_records = read_json(PROJDIR / "tests/expected_features_df_data.json")
    df = pd.DataFrame(expected_records["head"])
    df["naive_frequency_features_id"] = "foo"
    df["added_datetime"] = "foo"
    actual_x_data = sb.process_columns("naive_frequency_features", df)
    expected_x_data = df.drop(columns=non_feature_cols)

    diff = DeepDiff(expected_x_data, actual_x_data, significant_digits=3)

    assert "values_changed" not in diff, "values_changed"
    assert "iterable_item_added" not in diff, "iterable_item_added"
    assert "iterable_item_removed" not in diff, "iterable_item_removed"
    assert "dictionary_item_added" not in diff, "dictionary_item_added"
    assert "dictionary_item_removed" not in diff, "dictionary_item_removed"

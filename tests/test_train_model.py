import json
from pathlib import Path

import pandas as pd
from deepdiff import DeepDiff
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

import src.models.train_model as tm

PROJDIR = Path(__file__).resolve().parents[1]


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data


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
    actual_x_data = tm.process_columns("naive_frequency_features", df)
    expected_x_data = df.drop(columns=non_feature_cols)

    diff = DeepDiff(expected_x_data, actual_x_data, significant_digits=3)

    assert "values_changed" not in diff, "values_changed"
    assert "iterable_item_added" not in diff, "iterable_item_added"
    assert "iterable_item_removed" not in diff, "iterable_item_removed"
    assert "dictionary_item_added" not in diff, "dictionary_item_added"
    assert "dictionary_item_removed" not in diff, "dictionary_item_removed"


def test_get_classifier():
    params = {
        "n_iter_no_change": 50,
        "random_state": 42,
        "tol": 0.001,
        "warm_start": True,
    }
    actual_pipe = Pipeline(
        [("dv", DictVectorizer()), ("clf", GradientBoostingClassifier(**params))]
    )
    actual_dv = actual_pipe.get_params()["dv"]
    actual_clf = actual_pipe.get_params()["clf"]

    expected_pipe = tm._get_named_classifier("gradientboostingclassifier", params)
    expected_dv = expected_pipe.get_params()["dv"]
    expected_clf = expected_pipe.get_params()["clf"]

    assert expected_pipe.__repr__() == actual_pipe.__repr__(), "pipelines different"
    assert expected_dv.__repr__() == actual_dv.__repr__(), "dict vectorizers different"
    assert expected_clf.__repr__() == actual_clf.__repr__(), "classifiers different"
    assert (
        expected_clf.get_params() == actual_clf.get_params()
    ), "classifier parameters different"

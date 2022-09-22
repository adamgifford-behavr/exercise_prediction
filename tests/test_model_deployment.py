import base64
import json
import pickle
from pathlib import Path

from deepdiff import DeepDiff
from sklearn.base import ClassifierMixin

from src.deployment.streaming import model


class ModelMock(ClassifierMixin):
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = len(X)
        return [self.value] * n


def read_text(file):
    test_directory = Path(__file__).parent

    with open(test_directory / file, "rt", encoding="utf-8") as f_in:
        return f_in.read().strip()


def load_json(filepath: str) -> dict:
    """
    It opens the json file at the given filepath, reads the contents, and returns the
    contents as a dictionary

    Args:
      filepath (str): The path to the file you want to load.

    Returns:
      A dictionary
    """
    test_directory = Path(__file__).parent

    with open(test_directory / filepath, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data


frequency_features = load_json("frequency_features.json")
feature_cols = [
    f"{key}_{int(v_ix)}" for key, val in frequency_features.items() for v_ix in val
]
model_mock = ModelMock("Non-exercise")
model_version = "Test123"
model_service = model.ModelService(
    model_mock, frequency_features, feature_cols, model_version
)


def test_get_feature_col_names():
    example_freq_feats = load_json("example_frequency_features.json")
    actual_feature_cols = model.get_feature_col_names(example_freq_feats)
    expected_freq_feats = [
        "accel_x_0",
        "accel_x_1",
        "accel_x_2",
        "accel_x_4",
        "accel_x_5",
        "accel_x_7",
        "accel_x_9",
        "accel_x_12",
        "accel_x_16",
    ]
    assert expected_freq_feats == actual_feature_cols


def test_base64_decode():
    base64_input = read_text("data.b64")
    actual_result = model.base64_decode(base64_input)
    expected_result = load_json("example_decoded_data.json")

    diff = DeepDiff(expected_result, actual_result)
    print(f"diff={diff}")

    assert "values_changed" not in diff, "values_changed"
    assert "iterable_item_added" not in diff, "iterable_item_added"
    assert "iterable_item_removed" not in diff, "iterable_item_removed"
    assert "dictionary_item_added" not in diff, "dictionary_item_added"
    assert "dictionary_item_removed" not in diff, "dictionary_item_removed"


def test_make_feature_id():
    first_record = {"subject_id": 1, "file_id": 1, "data_id": 1, "time": 0}
    last_record = {"subject_id": 1, "file_id": 1, "data_id": 1, "time": 3}

    actual_result = model.make_feature_id(first_record, last_record)

    expected_result = "1_1_1_0_3"

    assert expected_result == actual_result


def test_prepare_data():
    records = load_json("example_decoded_data.json")
    actual_result = model_service.prepare_data(records)

    test_directory = Path(__file__).parent
    with open(test_directory / "example_prepared_data.p", "rb") as infile:
        expected_result = pickle.load(infile)

    assert not (expected_result - actual_result).any()


def test_prepare_features():
    test_directory = Path(__file__).parent
    with open(test_directory / "example_prepared_data.p", "rb") as infile:
        ndarray = pickle.load(infile)

    actual_result = model_service.prepare_features(ndarray)
    expected_result = load_json("example_features_sample.json")

    diff = DeepDiff(expected_result, actual_result)
    print(f"diff={diff}")

    assert "values_changed" not in diff, "values_changed"
    assert "iterable_item_added" not in diff, "iterable_item_added"
    assert "iterable_item_removed" not in diff, "iterable_item_removed"
    assert "dictionary_item_added" not in diff, "dictionary_item_added"
    assert "dictionary_item_removed" not in diff, "dictionary_item_removed"


def test_predict():
    X = load_json("example_features_sample.json")
    actual_prediction = model_service.predict(X)
    expected_prediction = "Non-exercise"

    assert expected_prediction == actual_prediction


def test_lambda_handler():
    base64_input = read_text("data.b64")
    decoded_input = model.base64_decode(base64_input)

    event = {
        "Records": [
            {
                "kinesis": {
                    "data": base64.b64encode(json.dumps(record).encode("utf-8")).decode(
                        "utf-8"
                    ),
                },
            }
            for record in decoded_input
        ]
    }
    actual_prediction = model_service.lambda_handler(event)

    prediction_event = {
        "model": "exercise_prediction_naive_feats_orch_cloud",
        "version": "Test123",
        "prediction": {
            "exercise": "Non-exercise",
            "id": "512_134_1_0.0_2.9999805304374165",
        },
    }

    diff = DeepDiff(prediction_event, actual_prediction)
    print(f"diff={diff}")

    assert "values_changed" not in diff, "values_changed"
    assert "iterable_item_added" not in diff, "iterable_item_added"
    assert "iterable_item_removed" not in diff, "iterable_item_removed"
    assert "dictionary_item_added" not in diff, "dictionary_item_added"
    assert "dictionary_item_removed" not in diff, "dictionary_item_removed"

# pylint: disable=duplicate-code

import base64
import json
from pathlib import Path

import requests  # type: ignore
from deepdiff import DeepDiff


def read_text(file):
    test_directory = Path(__file__).parent

    with open(test_directory / file, "rt", encoding="utf-8") as f_in:
        return f_in.read().strip()


def base64_decode(encoded_data):
    decoded_data = base64.b64decode(encoded_data).decode("utf-8")
    signals_event = json.loads(decoded_data)
    return signals_event


def base64_encode(decoded_data):
    encoded_data = base64.b64encode(json.dumps(decoded_data).encode("utf-8"))
    return encoded_data


base64_input = read_text("data.b64")
decoded_input = base64_decode(base64_input)

event = {
    "Records": [
        {
            "kinesis": {
                "data": base64_encode(record).decode("utf-8"),
            },
        }
        for record in decoded_input
    ]
}


url = "http://localhost:8080/2015-03-31/functions/function/invocations"
actual_response = requests.post(url, json=event).json()
print("actual response:")

print(json.dumps(actual_response, indent=2))

expected_response = {
    "model": "exercise_prediction_naive_feats_orch_cloud",
    "version": "85412c4643564a4f8a6b3d8f0130216a",
    "prediction": {
        "exercise": "Non-exercise",
        "id": "512_134_1_0.0_2.9999805304374165",
    },
}


diff = DeepDiff(expected_response, actual_response)
print(f"diff={diff}")

assert "values_changed" not in diff, "values_changed"
assert "iterable_item_added" not in diff, "iterable_item_added"
assert "iterable_item_removed" not in diff, "iterable_item_removed"
assert "dictionary_item_added" not in diff, "dictionary_item_added"
assert "dictionary_item_removed" not in diff, "dictionary_item_removed"

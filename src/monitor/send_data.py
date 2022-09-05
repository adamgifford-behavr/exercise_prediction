"""
This script simulates sending streaming data to the prediction service
"""
import json
from datetime import datetime
from time import sleep

import pyarrow.parquet as pq
import requests  # type: ignore

table = pq.read_table("preprocessed_fileID12_subjID10003_dataID0.parquet")
data = table.to_pylist()


class DateTimeEncoder(json.JSONEncoder):
    """It converts datetime objects to ISO format."""

    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


with open("target.csv", "w", encoding="utf-8") as f_target:
    for row in data:
        if not row["label_group"]:
            continue

        f_target.write(f"{row['naive_frequency_features_id']},{row['label_group']}\n")

        resp_data = requests.post(
            "http://127.0.0.1:9696/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(row, cls=DateTimeEncoder),
        )
        print(resp_data)
        resp = resp_data.json()
        print(f"prediction: {resp['prediction']}")
        sleep(3)  # data is preprocessed in 3-s windows

"""
This module is used to send a test packet of accelerometer and gyroscope data to our
prediction service and print out the predicted response.
"""

import json

import requests  # type: ignore

with open("sample_fileID134_subjID512_dataID1.json", "r", encoding="utf-8") as infile:
    sample = json.load(infile)

URL = "http://localhost:9696/predict"
response = requests.post(URL, json=sample, timeout=3)
print(response.json())

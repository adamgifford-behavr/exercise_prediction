"""Code to simulate streaming data to database for stream prediction of exercise"""
# -*- coding: utf-8 -*-
import json
import uuid
from time import sleep

from fastparquet import ParquetFile

DATA_SPLIT_JSON = "..\\features\\datafile_group_splits.json"
with open(DATA_SPLIT_JSON, "r", encoding="utf-8") as infile:
    data_splits = json.load(infile)

streaming_file = data_splits["simulate"]
df = ParquetFile(streaming_file).to_pandas()

# assuming 2 ms per iteration of for loop, this is 1/T to sleep between sending data
fs = 50 - 0.002
DT = 1 / fs

for ix, row in df.itterows():
    row["id"] = str(uuid.uuid4())
    # send row to SOMETHING

    sleep(DT)

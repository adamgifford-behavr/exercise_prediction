"""
This script tests the docker image for the AWS lambda function by sending "packets"
of simulation exercise data in real time and returning the predicted exercise.
"""
import base64
import json
from copy import deepcopy
from time import process_time, process_time_ns, sleep
from typing import TypedDict

import pyarrow.parquet as pq
import requests  # type: ignore
from numpy import random

KinesisRecord = TypedDict(
    "KinesisRecord",
    {
        "kinesisSchemaVersion": str,
        "partitionKey": str,
        "sequenceNumber": str,
        "data": str,
        "approximateArrivalTimestamp": float | int,
    },
)
BaseRecord = TypedDict(
    "BaseRecord",
    {
        "kinesis": KinesisRecord,
        "eventSource": str,
        "eventVersion": str,
        "eventID": str,
        "eventName": str,
        "invokeIdentityArn": str,
        "awsRegion": str,
        "eventSourceARN": str,
    },
)
base_record: BaseRecord = {
    "kinesis": {
        "kinesisSchemaVersion": "1.0",
        "partitionKey": "",
        "sequenceNumber": "",
        "data": "",
        "approximateArrivalTimestamp": 0,
    },
    "eventSource": "aws:kinesis",
    "eventVersion": "1.0",
    "eventID": "shardId-000000000000:",
    "eventName": "aws:kinesis:record",
    "invokeIdentityArn": "arn:aws:iam::849352486600:role/lambda-kinesis-role",
    "awsRegion": "us-east-1",
    "eventSourceARN": "arn:aws:kinesis:us-east-1:849352486600:stream/signals_stream",
}
BASE_SEQUENCE_N = 49630081666084879290581185630324770398608704880802529282
URL = "http://localhost:8080/2015-03-31/functions/function/invocations"


print("loading simulated streaming data")
table = pq.read_table("../../../data/interim/fileID12_subjID10003_dataID0.parquet")
sim_data = table.to_pylist()
NSAMPLES = len(sim_data)

print("sending data...")
ix = 0
mu, sigma = 151, 5
stream_sample_n = round(random.normal(mu, sigma))  # type: ignore
T_ELAPSED = 0
while True:
    t_start = process_time()
    if ix + stream_sample_n > NSAMPLES:
        break

    stream_data = sim_data[ix : (ix + stream_sample_n)]
    records = []
    for row in stream_data:
        PARTITION_KEY = str(row["subject_id"])
        SEQUENCE_NUMBER = str(BASE_SEQUENCE_N + int(T_ELAPSED))
        data = base64.b64encode(json.dumps(row).encode("utf-8")).decode("utf-8")

        record = deepcopy(base_record)
        record["kinesis"]["partitionKey"] = PARTITION_KEY
        record["kinesis"]["sequenceNumber"] = SEQUENCE_NUMBER
        record["kinesis"]["data"] = data
        record["kinesis"]["approximateArrivalTimestamp"] = process_time_ns()

        record["eventID"] = record["eventID"] + SEQUENCE_NUMBER
        records.append(record)

    event = {"Records": records}
    response = requests.post(URL, json=event, timeout=3)
    print(response.json())

    ix += stream_sample_n
    stream_sample_n = round(random.normal(mu, sigma))
    T_ELAPSED = process_time() - t_start  # type: ignore
    sleep(3 - T_ELAPSED)

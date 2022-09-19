"""
This script tests the implementation of the lambda function in AWS putting simulation
exercise data into an Amazon Kinesis stream for raw signal data, which acts as the trigger
for the lambda function. Subsequently, the script pings the output stream to get the
associated model predictions.
"""
import json
from datetime import datetime
from time import sleep

import boto3
import pyarrow.parquet as pq

client = boto3.client("kinesis")

print("loading simulated streaming data")
table = pq.read_table("../../../data/interim/fileID12_subjID10003_dataID0.parquet")
data = table.to_pylist()
print("sending data...")
now = datetime.now()
for ix, row in enumerate(data):
    if ix > 150:  # just sending ~1 3-s window worth of data for prediction
        break

    response = client.put_record(
        StreamName="signals-stream",
        Data=json.dumps(row),
        PartitionKey=str(row["subject_id"]),
    )
    sleep(0.015)

shard_iterator = client.get_shard_iterator(
    StreamName="predictions-stream",
    ShardId="shardId-000000000000",
    ShardIteratorType="AT_TIMESTAMP",
    Timestamp=now,
)["ShardIterator"]

NEXT_SHARD_ITERATOR = True
while NEXT_SHARD_ITERATOR:
    predictions = client.get_records(ShardIterator=shard_iterator, Limit=100)

    for record in predictions["Records"]:
        data = json.loads(record["Data"])
        print(data)

    NEXT_SHARD_ITERATOR = predictions["NextShardIterator"]

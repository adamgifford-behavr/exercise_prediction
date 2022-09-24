#!/usr/bin/env bash

export KINESIS_STREAM_INPUT="stg_signals_stream-exercise-prediction"
export KINESIS_STREAM_OUTPUT="stg_predictions_stream-exercise-prediction"

SHARD_ID=$(aws kinesis put-record  \
        --stream-name ${KINESIS_STREAM_INPUT}   \
        --partition-key 1  --cli-binary-format raw-in-base64-out  \
        --data   '{"accel_x": -0.7877657768725429,
            "accel_y": -0.3509873035097201,
            "accel_z": 0.49528825563794754,
            "data_id": 1,
            "file_id": 134,
            "gyro_x": 13.832166585813447,
            "gyro_y": 4.861046094025692,
            "gyro_z": -12.426823304809789,
            "label": "<Initial Activity>",
            "label_group": "Junk",
            "subject_id": 512,
            "time": 0.0
        }',  \
        --query 'ShardId' \
        | tr -d '"'
)

SHARD_ITERATOR=$(aws kinesis get-shard-iterator --shard-id ${SHARD_ID} --shard-iterator-type TRIM_HORIZON --stream-name ${KINESIS_STREAM_OUTPUT} --query 'ShardIterator')

aws kinesis get-records --shard-iterator $SHARD_ITERATOR
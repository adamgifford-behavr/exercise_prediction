"""This module is the lambda function that would be used in AWS lambda for a true
streaming service implementation of the model. Here, it is used in a Docker container
image replicating the AWS lambda service to simulate a streaming prediction service.
"""
import base64
import json
import os

import boto3
import mlflow
import numpy as np
from aws_lambda_typing import context as context_
from aws_lambda_typing import events

kinesis_client = boto3.client("kinesis")

TEST_RUN = os.getenv("TEST_RUN", "False") == "True"
PREDICTIONS_STREAM_NAME = os.getenv("PREDICTIONS_STREAM_NAME")

with open("frequency_features.json", "r", encoding="utf-8") as infile:
    DESIRED_FREQS = json.load(infile)

FEATURE_COLS = [
    f"{key}_{int(v_ix)}" for key, val in DESIRED_FREQS.items() for v_ix in val
]

DROP_COLS = [
    "time",
    "file_id",
    "subject_id",
    "data_id",
    "label",
    "label_group",
]

DT = np.dtype(
    [
        ("accel_x", float),
        ("accel_y", float),
        ("accel_z", float),
        ("gyro_x", float),
        ("gyro_y", float),
        ("gyro_z", float),
    ]
)

logged_model = os.getenv("MODEL_LOCATION")
model = mlflow.pyfunc.load_model(logged_model)


def _generate_window_fxn(n_fft: int, n_data: int) -> np.ndarray:
    """
    It creates a matrix of Hann windows, where each column is a Hann window, the number of
    rows is equal to the number of data points, and the number of columns is equal to
    the number of different signals.

    Args:
      n_fft (int): The number of samples in each FFT.
      n_data (int): number of recording signals

    Returns:
      A 2D array of size (n_fft, n_data)
    """
    return np.tile(np.hanning(n_fft), (n_data, 1)).T


def _calculate_frequencies(n_fft: int) -> np.ndarray:
    """
    It calculates the frequency vector for a given FFT length and sampling frequency

    Args:
      n_fft (int): The number of points in the FFT.

    Returns:
      The frequencies of the FFT.
    """
    n_points = 2 * int(np.floor(n_fft / 2))
    if n_fft % 2:
        n_points += 1
    freq = 50 / 2 * np.linspace(-1, 1, n_points)  # 50 is sampling frequency
    return freq


def _get_frequencies_indices(all_freqs: np.ndarray, desired_freqs: list) -> np.ndarray:
    """
    It takes a list of all frequencies and a list of desired frequencies, and returns the
    indices of the closest frequencies in the list of all frequencies

    Args:
      all_freqs (np.ndarray): all frequencies of the FFT to be computed
      desired_freqs (list): the frequencies you want to extract from the data

    Returns:
      The indices of the closest frequencies in the all_freqs array to the desired_freqs
    array.
    """
    closest_freqs_ix = np.array(
        [(np.abs([af - df for af in all_freqs])).argmin() for df in desired_freqs]
    )
    return closest_freqs_ix


def _calculate_spectrum(ndarray: np.ndarray) -> np.ndarray:
    """
    It takes a 2D numpy array, performs a 1D FFT on each row, and then normalizes the result

    Args:
      ndarray (np.ndarray): the input data

    Returns:
      The return value is a numpy array of the same shape as the input array.
    """
    X_w = np.fft.fftshift(np.fft.fft(ndarray, axis=0), axes=0)
    return np.abs(X_w / np.tile(abs(X_w).max(axis=0), (ndarray.shape[0], 1)))


def _calculate_single_window_features(X_w: np.ndarray, freq_ixs: list) -> list:
    """
    It takes a windowed data spectrum across raw measurements and a list of frequency
    indices, and returns a flattened list of features for each measurement and desired
    frequency in the window

    Args:
      X_w (np.ndarray): the windowed data in the shape of n_frequencies x n_columns
      freq_ixs (list): the indices of the frequencies we want to use for each
      measurement

    Returns:
      A list of features for each window.
    """
    # get only the desired frequencies by measurement
    meas_feats = [X_w[f_ix, ix] for ix, f_ix in enumerate(freq_ixs)]
    # flatten the features to have same shape as feat_cols
    flat_feats = [f_data for col in meas_feats for f_data in col]
    return flat_feats


def prepare_data(records: list[dict]) -> np.ndarray:
    """
    It takes a list of dictionaries, and returns a numpy array (after removing non-data
    fields).

    Args:
      records (dict): dictionary of raw data records

    Returns:
      A numpy array of the data records.
    """
    data_records = [
        {key: val for key, val in record.items() if key not in DROP_COLS}
        for record in records
    ]
    ndarray = np.zeros((len(data_records), len(data_records[0].keys())), dtype=DT)
    ndarray = np.array([list(record.values()) for record in data_records])
    return ndarray


def prepare_features(ndarray: np.ndarray) -> list[dict]:
    """
    We take the raw formatted data, apply a window function, calculate the spectrum, and
    then calculate the features

    Args:
      ndarray (np.ndarray): raw data records converted to a numpy array

    Returns:
      A list of dictionaries.
    """
    n_fft, n_data = ndarray.shape
    window = _generate_window_fxn(n_fft, n_data)
    freqs = _calculate_frequencies(n_fft)
    freq_ixs = [_get_frequencies_indices(freqs, val) for val in DESIRED_FREQS.values()]

    X_w = _calculate_spectrum(window * ndarray)
    flat_raw_feats = _calculate_single_window_features(X_w, freq_ixs)
    X = [dict(zip(FEATURE_COLS, flat_raw_feats))]
    return X


def predict(X: list[dict]) -> str:
    """
    It takes a list of dictionaries, each of which represents a single featurized
    observation, and returns a string representing the predicted class for each observation

    Args:
      X (list[dict]): a list of dictionaries of featurized observations

    Returns:
      The prediction of the model.
    """
    prediction = model.predict(X)
    return str(prediction[0])


def lambda_handler(
    event: events.SQSEvent, context: context_.Context  # pylint: disable=unused-argument
) -> dict:
    """
    It takes in a dictionary of records coming from an Amazon Kinesis stream, with each
    record containing a data field of encoded data representing accerlerometer, gyroscope,
    and subject metadata for a single time step, and returns a prediction of the exercise
    being performed

    Args:
      event (events.SQSEvent): the event of records that triggered the lambda function
      context (context_.Context): The AWS Lambda runtime passes this object to the handler.
      You can use it to access runtime methods, such as logging.

    Returns:
      The prediction event is being returned.
    """
    records = [
        json.loads(base64.b64decode(record["kinesis"]["data"]).decode("utf-8"))
        for record in event["Records"]
    ]

    sid = records[0]["subject_id"]
    fid = records[0]["file_id"]
    did = records[0]["data_id"]
    tid0, tid1 = (
        records[0]["time"],
        records[-1]["time"],
    )
    id_ = f"{sid}_{fid}_{did}_{tid0}_{tid1}"

    ndarray = prepare_data(records)
    X = prepare_features(ndarray)
    prediction = predict(X)

    prediction_event = {
        "model": "exercise_prediction_naive_feats_orch_cloud",
        "version": "1",
        "prediction": {"exercise": prediction, "id": id_},
    }

    if not TEST_RUN:
        kinesis_client.put_record(
            StreamName=PREDICTIONS_STREAM_NAME,
            Data=json.dumps(prediction_event),
            PartitionKey=str(records[0]["subject_id"]),
        )

    return prediction_event

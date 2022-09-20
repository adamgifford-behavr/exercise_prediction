# pylint: disable=duplicate-code
# pylint: disable=import-error
"""This module converts the lambda function that would be used in AWS lambda for a true
streaming service implementation into a model service class. It is used in the Docker
container image replicating the AWS lambda service to simulate a streaming prediction
service.
"""
import base64
import json
import os
from typing import Optional, Union

import boto3
import mlflow
import numpy as np
from aws_lambda_typing import events
from mypy_boto3_kinesis import KinesisClient
from sklearn.base import ClassifierMixin

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


def load_json(filepath: str) -> dict:
    """
    It opens the json file at the given filepath, reads the contents, and returns the
    contents as a dictionary

    Args:
      filepath (str): The path to the file you want to load.

    Returns:
      A dictionary
    """
    with open(filepath, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data


def get_feature_col_names(desired_freqs: dict) -> list:
    """
    Returns the list of feature column names in the form of <SIGNAL>_<FREQUNECY>, where
    <SIGNAL> is the name of the recording signal (e.g., "accel_x") and <FREQUENCY> is the
    frequency feature of interest for that signal (e.g., "0" for 0 Hz).

    Args:
      desired_freqs (dict): dictionary of frequency features, where keys are the signals
      and values are the lists of frequency features per signal

    Returns:
      A list of feature column names.
    """
    feature_cols = [
        f"{key}_{int(v_ix)}" for key, val in desired_freqs.items() for v_ix in val
    ]
    return feature_cols


def get_model_location(run_id: str) -> str:
    """
    If the `MODEL_LOCATION` environment variable is set, return it; otherwise, return the
    model location in S3

    Args:
      run_id (str): The run ID of the model you want to load.

    Returns:
      The model location is being returned.
    """

    model_bucket = os.getenv("MODEL_BUCKET")
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    if model_bucket and experiment_id:
        model_location = (
            f"s3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models/"
        )
    else:
        model_location = os.getenv("MODEL_LOCATION", "models/")

    return model_location


def load_model(run_id: str) -> ClassifierMixin:
    """
    It loads the model from the specified run ID and returns it

    Args:
      run_id (str): The run ID of the model you want to load.

    Returns:
      A ClassifierMixin object
    """
    model_path = get_model_location(run_id)
    model = mlflow.pyfunc.load_model(model_path)
    return model


def base64_decode(encoded_data: bytes) -> dict:
    """
    It takes a base64 encoded bytes string, decodes it, and returns the decoded data
    (which takes the format of a dictionary).

    Args:
      encoded_data (bytes): The base64 encoded data from the event.

    Returns:
      A dictionary
    """
    decoded_data = base64.b64decode(encoded_data).decode("utf-8")
    signals_event = json.loads(decoded_data)
    return signals_event


def make_feature_id(first_record: dict, last_record: dict) -> str:
    """
    It takes the first and last records of a feature and returns a string that uniquely
    identifies that feature that is generated for this packet of raw signals data

    Args:
      first_record (dict): the first record in the window
      last_record (dict): the last record in the window

    Returns:
      A string with the subject_id, file_id, data_id, and time range.
    """
    sid = first_record["subject_id"]
    fid = first_record["file_id"]
    did = first_record["data_id"]
    tid0, tid1 = first_record["time"], last_record["time"]

    id_ = f"{sid}_{fid}_{did}_{tid0}_{tid1}"
    return id_


class ModelService:
    """This class serves as the model deployment for our streaming service with AWS
    lambda and allows for optional callbacks to put prediction records into a Kinesis
    stream
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model: ClassifierMixin,
        desired_freqs: dict,
        feature_cols: list,
        model_version: Optional[str] = None,
        callbacks: Optional[list] = None,
    ):
        """
        This function takes in a model, a dictionary of desired frequency features, a list
        of feature column names, a model version, and a list of callbacks.

        Args:
          model (ClassifierMixin): The model to use for deployment.
          desired_freqs (dict): a dictionary of the desired frequencies for each recording
          signal used in the model.
          feature_cols (list): The names of the feature columns of the resulting feaures.
          model_version (Optional[str]): This is the version of the model that you're
          deploying
          callbacks (Optional[list]): A list of callbacks to be used during predictions.
        """
        self.model = model
        self.desired_freqs = desired_freqs
        self.feature_cols = feature_cols
        self.model_version = model_version
        self.callbacks = callbacks or []

    @staticmethod
    def _generate_window_fxn(n_fft: int, n_data: int) -> np.ndarray:
        """
        It creates a matrix of Hann windows, where each column is a Hann window, the number
        of rows is equal to the number of data points, and the number of columns is equal
        to the number of different signals.

        Args:
        n_fft (int): The number of samples in each FFT.
        n_data (int): number of recording signals

        Returns:
        A 2D array of size (n_fft, n_data)
        """
        return np.tile(np.hanning(n_fft), (n_data, 1)).T

    @staticmethod
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

    @staticmethod
    def _get_frequencies_indices(
        all_freqs: np.ndarray, desired_freqs: list
    ) -> np.ndarray:
        """
        It takes a list of all frequencies and a list of desired frequencies, and returns
        the indices of the closest frequencies in the list of all frequencies

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

    @staticmethod
    def _calculate_spectrum(ndarray: np.ndarray) -> np.ndarray:
        """
        It takes a 2D numpy array, performs a 1D FFT on each row, and then normalizes the
        result

        Args:
        ndarray (np.ndarray): the input data

        Returns:
        The return value is a numpy array of the same shape as the input array.
        """
        X_w = np.fft.fftshift(np.fft.fft(ndarray, axis=0), axes=0)
        return np.abs(X_w / np.tile(abs(X_w).max(axis=0), (ndarray.shape[0], 1)))

    @staticmethod
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

    @staticmethod
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

    def prepare_features(self, ndarray: np.ndarray) -> list[dict]:
        """
        We take the raw formatted data, apply a window function, calculate the spectrum,
        and then calculate the features

        Args:
        ndarray (np.ndarray): raw data records converted to a numpy array

        Returns:
        A list of dictionaries.
        """
        n_fft, n_data = ndarray.shape
        window = self._generate_window_fxn(n_fft, n_data)
        freqs = self._calculate_frequencies(n_fft)
        freq_ixs = [
            self._get_frequencies_indices(freqs, val)
            for val in self.desired_freqs.values()
        ]

        X_w = self._calculate_spectrum(window * ndarray)
        flat_raw_feats = self._calculate_single_window_features(X_w, freq_ixs)
        X = [dict(zip(self.feature_cols, flat_raw_feats))]
        return X

    def predict(self, X: list[dict]) -> str:
        """
        It takes a list of dictionaries, each of which represents a single featurized
        observation, and returns a string representing the predicted class for each
        observation

        Args:
        X (list[dict]): a list of dictionaries of featurized observations

        Returns:
        The prediction of the model.
        """
        prediction = self.model.predict(X)
        return str(prediction[0])

    def lambda_handler(self, event: events.SQSEvent) -> dict:
        """
        It takes in a dictionary of records coming from an Amazon Kinesis stream, with
        each record containing a data field of encoded data representing accerlerometer,
        gyroscope, and subject metadata for a single time step, and returns a prediction
        of the exercise being performed

        Args:
        event (events.SQSEvent): the event of records that triggered the lambda function

        Returns:
        The prediction event is being returned.
        """
        records = [
            base64_decode(record["kinesis"]["data"]) for record in event["Records"]
        ]

        id_ = make_feature_id(records[0], records[-1])

        ndarray = self.prepare_data(records)
        X = self.prepare_features(ndarray)
        prediction = self.predict(X)

        prediction_event = {
            "model": "exercise_prediction_naive_feats_orch_cloud",
            "version": "1",
            "prediction": {"exercise": prediction, "id": id_},
        }

        for callback in self.callbacks:
            callback(prediction_event)

        return prediction_event


# It takes a Kinesis client and a stream name as input, and then it has a function that
# takes a prediction event as input and puts it into the Kinesis stream
class KinesisCallback:  # pylint: disable=too-few-public-methods
    """
    Callback class that takes a Kinesis client and a stream name as input and has a
    function that takes a prediction event as input and puts it into the Kinesis stream
    """

    def __init__(
        self, kinesis_client: KinesisClient, prediction_stream_name: str
    ) -> None:
        """
        This function initializes the class with a Kinesis client and the name of the
        stream to which the predictions will be sent.

        Args:
          kinesis_client (KinesisClient): The Kinesis client object.
          prediction_stream_name (str): The name of the Kinesis stream that will be used
          to send the predictions to.
        """
        self.kinesis_client = kinesis_client
        self.prediction_stream_name = prediction_stream_name

    def put_record(self, prediction_event: dict) -> None:
        """
        The function takes a prediction event as input, and then puts the event into the
        Kinesis stream

        Args:
          prediction_event (dict): The event that we want to put into the stream.
        """
        id_ = prediction_event["prediction"]["id"]

        self.kinesis_client.put_record(
            StreamName=self.prediction_stream_name,
            Data=json.dumps(prediction_event),
            PartitionKey=str(id_),
        )


def create_kinesis_client() -> KinesisClient:
    """
    It creates a Kinesis client

    Returns:
      A KinesisClient object
    """
    endpoint_url = os.getenv("KINESIS_ENDPOINT_URL")

    if endpoint_url is None:
        return boto3.client("kinesis")

    return boto3.client("kinesis", endpoint_url=endpoint_url)


def init(
    prediction_stream_name: str,
    desired_freqs: Union[str, dict],
    run_id: str,
    test_run: bool,
) -> ModelService:
    """
    It loads the model, creates a `ModelService` object for streaming service with AWS,
    and returns it

    Args:
      prediction_stream_name (str): The name of the Kinesis stream to which we'll send
      predictions.
      desired_freqs (Union[str, dict]): This is either the path to a json file or a dictionary
      of the frequency features used to build the model. The keys are the signal recordings
      and the values frequency features for that signal.
      run_id (str): The run_id of the model you want to deploy.
      test_run (bool): If True, the model will not send predictions to Kinesis. This is
      useful for testing.

    Returns:
      A ModelService object
    """
    if isinstance(desired_freqs, str):
        desired_freqs = load_json(desired_freqs)

    feature_cols = get_feature_col_names(desired_freqs)
    model = load_model(run_id)

    callbacks = []

    if not test_run:
        kinesis_client = create_kinesis_client()
        kinesis_callback = KinesisCallback(kinesis_client, prediction_stream_name)
        callbacks.append(kinesis_callback.put_record)

    model_service = ModelService(
        model=model,
        desired_freqs=desired_freqs,
        feature_cols=feature_cols,
        model_version=run_id,
        callbacks=callbacks,
    )

    return model_service

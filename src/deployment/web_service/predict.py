"""Prediction app for web-service model prediction"""
import json
import os

import mlflow
import numpy as np
from flask import Flask, jsonify, request
from scipy import signal
from scipy.fftpack import fft, fftshift

logged_model = os.getenv("MODEL_LOCATION", "models/")
model = mlflow.pyfunc.load_model(logged_model)

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

app = Flask("exercise_predictions")


def _generate_window_fxn(n_fft, n_data):
    return np.tile(signal.hann(n_fft), (n_data, 1)).T


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
    X_w = fftshift(fft(ndarray, axis=0), axes=0)
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


def prepare_data(records):
    """prepares the data for prediction"""
    data_records = [
        {key: val for key, val in record.items() if key not in DROP_COLS}
        for record in records
    ]
    ndarray = np.zeros((len(data_records), len(data_records[0].keys())), dtype=DT)
    ndarray = np.array([list(record.values()) for record in data_records])
    return ndarray


def prepare_features(ndarray):
    "Creates the features from raw data signals"
    n_fft, n_data = ndarray.shape
    window = _generate_window_fxn(n_fft, n_data)
    freqs = _calculate_frequencies(n_fft)
    freq_ixs = [_get_frequencies_indices(freqs, val) for val in DESIRED_FREQS.values()]

    X_w = _calculate_spectrum(window * ndarray)
    flat_raw_feats = _calculate_single_window_features(X_w, freq_ixs)
    X = [dict(zip(FEATURE_COLS, flat_raw_feats))]
    return X


def predict(X):
    "Generates prediction"
    prediction = model.predict(X)
    return str(prediction[0])


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    "Web service endpoint to receive raw data for prediction"
    records = request.get_json()

    ndarray = prepare_data(records)
    X = prepare_features(ndarray)
    prediction = predict(X)

    result = {"prediction": prediction}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)  # nosec

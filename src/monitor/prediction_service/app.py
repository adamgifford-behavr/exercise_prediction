"""
Flask app that acts as a prediction service for simulated streaming exercise data. It
takes a JSON object, processes it, makes a prediction, saves the prediction to a
database, and sends the prediction to another service

Args:
    record (dict): the record that was just processed

Returns:
    The result is a JSON object with the prediction.
"""
import os
import pickle  # nosec

import requests
from flask import Flask, jsonify, request
from flask.wrappers import Response
from pymongo import MongoClient

MODEL_FILE = os.getenv("MODEL_FILE", "model.pkl")

EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

with open(MODEL_FILE, "rb") as f_in:
    model = pickle.load(f_in)  # nosec


app = Flask("ex_pred_lblgrp_stream")
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")


def process_record(record: dict) -> dict:
    """
    It takes a dictionary of streamed data and returns a dictionary with the same keys
    but with the values of the keys that are in the list `non_feature_cols` removed

    Args:
      record (dict): dict

    Returns:
      A dictionary with the keys being the column names and the values being the values of
      the columns.
    """
    non_feature_cols = [
        "naive_frequency_features_id",
        "featurize_id",
        "file",
        "dataset_group",
        "added_datetime",
        "window_size",
        "t_index",
        "label",
        "label_group",
    ]
    clean_record = {
        key: val for key, val in record.items() if key not in non_feature_cols
    }
    return clean_record


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    """
    It takes a JSON object, processes it, makes a prediction, saves the prediction to a
    database, and sends the prediction to another service

    Returns:
      The result is a JSON object with the prediction.
    """
    record = request.get_json()

    clean_record = process_record(record)
    y_pred = str(model.predict([clean_record])[0])

    result = {
        "prediction": y_pred,
    }

    save_to_db(record, y_pred)
    send_to_evidently_service(record, y_pred)
    return jsonify(result)


def save_to_db(record: dict, prediction: str) -> None:
    """
    It takes a record and a prediction, and saves them to the database

    Args:
      record (dict): the record that was just processed
      prediction (str): The prediction that the model made
    """
    rec = record.copy()
    rec["prediction"] = prediction
    collection.insert_one(rec)


def send_to_evidently_service(record: dict, prediction: str) -> None:
    """
    It takes a record and a prediction, and sends them to the Evidently service

    Args:
      record (dict): the record that was just processed
      prediction (str): the prediction that you want to send to the Evidently service
    """
    rec = record.copy()
    rec["prediction"] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/exercise_group", json=[rec])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696)  # nosec

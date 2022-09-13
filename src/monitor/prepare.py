"""
This script prepares the data for the monitoring service by creating a dataframe of
streaming data from the database table, then saving it as a parquet file, and copying the
file to the evidently monitoring service directory.
"""
import os
from pathlib import Path

import mlflow
from dotenv import find_dotenv, load_dotenv

import src.models.score_batch as sb

load_dotenv(find_dotenv())
FEATURE_STORE_URI = os.getenv("FEATURE_STORE_URI", "localhost")
FEATURE_STORE_PW = os.getenv("FEATURE_STORE_PW")
DATABASE_URI = f"postgresql+psycopg2://postgres:{FEATURE_STORE_PW}@{FEATURE_STORE_URI}"
sb.DATABASE_URI = DATABASE_URI
sb.FEATURIZE_ID = os.getenv("FEATURIZE_ID")

sim_df = sb.load_data("naive_frequency_features", "simulate")
train_df = sb.load_data("naive_frequency_features", "train", limit=sim_df.shape[0])

stream_data_path = Path(sim_df.loc[0, "file"])
stream_file_name = "preprocessed_" + stream_data_path.parts[-1]
REF_FILE_NAME = "preprocessed_reference_data.parquet"

logged_model = os.getenv("MODEL_LOCATION", "prediction_service/models/")
model = mlflow.pyfunc.load_model(logged_model)

train_df = train_df.drop(
    columns=[
        "naive_frequency_features_id",
        "featurize_id",
        "file",
        "dataset_group",
        "added_datetime",
        "window_size",
        "t_index",
        "label",
    ]
)

sim_df = sim_df.drop(
    columns=[
        "naive_frequency_features_id",
        "featurize_id",
        "file",
        "dataset_group",
        "added_datetime",
        "window_size",
        "t_index",
        "label",
    ]
)

train_df["prediction"] = model.predict(
    train_df.drop(columns=["label_group"]).to_dict("records")
)
REF_PATH = "./evidently_service/datasets/"
STREAM_PATH = "./"

train_df.to_parquet(REF_PATH + REF_FILE_NAME, engine="pyarrow")
sim_df.to_parquet(STREAM_PATH + stream_file_name, engine="pyarrow")

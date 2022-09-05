"""
This script prepares the data for the monitoring service by creating a dataframe of
streaming data from the database table, then saving it as a parquet file, and copying the
file to the evidently monitoring service directory.
"""
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

import src.models.score_batch as sb

load_dotenv(find_dotenv())
FEATURE_STORE_URI = os.getenv("FEATURE_STORE_URI", "localhost")
FEATURE_STORE_PW = os.getenv("FEATURE_STORE_PW")
DATABASE_URI = (
    f"postgresql+psycopg2://postgres:{FEATURE_STORE_PW}@{FEATURE_STORE_URI}"
    "/feature_store"
)
sb.DATABASE_URI = DATABASE_URI
sb.FEATURIZE_ID = os.getenv("FEATURIZE_ID")

sim_df = sb.load_data("naive_frequency_features", "simulate")
dst_paths = ("./", "./evidently_service/datasets/")

file_path = Path(sim_df.loc[0, "file"])
file_name = "preprocessed_" + file_path.parts[-1]

for dst in dst_paths:
    # ran into issues with fastparquet, but pyarrow seems to work...
    sim_df.to_parquet(dst + file_name, engine="pyarrow")

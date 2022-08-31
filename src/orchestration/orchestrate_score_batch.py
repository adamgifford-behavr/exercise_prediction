# -*- coding: utf-8 -*-
"""
This module defines the Prefect flow for orchestrating batch model scoring, using the
functions and methods defined in src.models.score_batch.
"""
import os

import sqlalchemy as sa
from dotenv import find_dotenv, load_dotenv
from prefect import flow, get_run_logger, task
from prefect.task_runners import SequentialTaskRunner

import src.models.score_batch as sb

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())
FEATURE_STORE_URI = os.getenv("FEATURE_STORE_URI", "localhost")
FEATURE_STORE_PW = os.getenv("FEATURE_STORE_PW")
DATABASE_URI = (
    f"postgresql+psycopg2://postgres:{FEATURE_STORE_PW}@{FEATURE_STORE_URI}"
    "/feature_store"
)

MLFLOW_DB_URI = os.getenv("MLFLOW_DB_URI", "localhost:5000")
MLFLOW_DB_PW = os.getenv("MLFLOW_DB_PW")

load_data = task(sb.load_data, name="Load batch data")
apply_model = task(sb.apply_model, name="Apply model")
write_predictions_to_db = task(
    sb.write_predictions_to_db, name="Write predictions to database"
)


@flow(name="Model Batch Scoring", task_runner=SequentialTaskRunner())
def score_flow(
    feature_table: str = "naive_frequency_features",
    prediction_table: str = "naive_frequency_features_predictions",
    label_col: str = "label_group",
    model_name: str = "exercise_prediction_naive_feats_pipe",
    model_stage: str = "Production",
) -> None:
    """
    It loads data from `feature_table`, applies a model to the data, and writes the
    predictions to a `prediction_table`.

    Args:
      feature_table (str): The name of the table that contains the features that you want to
      use to make predictions. Defaults to naive_frequency_features
      prediction_table (str): The name of the table where the predictions will be written.
      Defaults to naive_frequency_features_predictions
      label_col (str): The name of the column in the feature table that contains the label
      to be predicted. Defaults to label_group
      model_name (str): The name of the MLflow registered model you want to use. Defaults
      to exercise_prediction_naive_feats_pipe
      model_stage (str): The stage of the registered model in the registry.
      Defaults to Production
    """
    logger = get_run_logger()

    logger.info("loading test data...")
    df_test_meta = load_data(feature_table, "simulate")
    logger.info("loading complete")

    logger.info("applying model to batch data...")
    model_data = {"name": model_name, "stage": model_stage}
    pred_df = apply_model(model_data, df_test_meta, feature_table, label_col)
    logger.info("model applied")

    logger.info("writing predictions to %s...", prediction_table)
    engine: sa.engine.base.Engine = sa.create_engine(
        DATABASE_URI,
        executemany_mode="values",
        executemany_values_page_size=10000,
        executemany_batch_page_size=500,
    )
    metadata = sa.schema.MetaData(bind=engine)
    table = sb._get_table_model(  # pylint: disable=protected-access
        engine, metadata, prediction_table
    )
    write_predictions_to_db(pred_df, table, engine)
    logger.info("writing complete")
    logger.info("complete")

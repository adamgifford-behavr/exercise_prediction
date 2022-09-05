"""
Module loads simulated batch data from a table in the database, applies a model to it, and
writes the predictions to a table in the database
"""
import logging
import os
from contextlib import contextmanager
from time import sleep
from typing import Any, Dict, Generator, Optional, Tuple, Union

import click
import mlflow
import pandas as pd
import sqlalchemy as sa
from dotenv import find_dotenv, load_dotenv
from mlflow.tracking import MlflowClient
from numpy import ndarray
from sqlalchemy.exc import NoSuchTableError
from sqlalchemy.orm import sessionmaker

from src.models.predictions_tables import NaiveFreqFeatsPred

TABLE_DICT = {"naive_frequency_features_predictions": NaiveFreqFeatsPred}


# def _read_json(file_path: Union[str, Path]) -> dict:
#     """
#     It reads a JSON file and returns the data as a dictionary

#     Args:
#       file_path (Union[str, Path]): The path to the JSON file.

#     Returns:
#       A dictionary
#     """
#     with open(file_path, "r", encoding="utf-8") as infile:
#         data = json.load(infile)
#     return data


@contextmanager
def _session_scope(
    engine: sa.engine.base.Engine,
) -> Generator[sa.orm.session.Session, Any, None]:
    """
    It creates a session, yields it, commits it, and closes it

    Args:
      engine (sa.engine.base.Engine): The engine to use for the session.
    """
    session = sessionmaker(bind=engine)()
    try:
        yield session
        session.commit()
    except:  # noqa: E722
        session.rollback()
        raise
    finally:
        session.close()


def _get_database_tools(
    table_name: str,
) -> Tuple[sa.engine.base.Engine, sa.sql.schema.Table]:
    """
    It creates a database engine and a table object from a table name

    Args:
      table_name (str): The name of the table you want to query.

    Returns:
      A tuple of the engine and the table.
    """
    engine = sa.create_engine(DATABASE_URI)
    metadata = sa.schema.MetaData(bind=engine)
    table = sa.Table(table_name, metadata, autoload=True)
    return engine, table


def load_data(
    table_name: str,
    data_group: str,
    limit: Optional[int] = None,
):
    """
    It takes a table name, a data group, and an optional limit, and returns a pandas
    dataframe

    Args:
      table_name (str): The name of the table in the database that you want to load data
      from.
      data_group (str): Which group of data you want
      load from the database. Options are "train" for training data, "validation" for
      validation data, or "test" for test data (or "simulate" for simulated batch data)
      limit (Optional[int]): Limit on the number of rows to load. Defaults to None (no
      limit)

    Returns:
      A dataframe
    """
    engine, table = _get_database_tools(table_name)

    with _session_scope(engine) as session:
        results = (
            session.query(table)
            .filter(
                sa.and_(
                    table.c.featurize_id == FEATURIZE_ID,
                    table.c.dataset_group == data_group,
                )
            )
            .limit(limit)
        )

    df = pd.read_sql(results.statement, con=engine, parse_dates=["added_datetime"])
    return df


def process_columns(
    table_name: str, df: pd.DataFrame, drop_additional_features: Optional[list] = None
) -> pd.DataFrame:
    """
    Drop columns that are not features, with additional option to drop a subset of feature
    columns. Finally, ensure that all column names are strings.

    Args:
      table_name (str): The name of the table where feature dataset comes from (needed
      to specify the primary key column).
      df (pd.DataFrame): the dataframe of features
      drop_additional_features (Optional[list]): list of additional features to drop
      from the dataframe

    Returns:
      The dataframe with the columns dropped.
    """
    non_feature_cols = [
        table_name + "_id",
        "featurize_id",
        "file",
        "dataset_group",
        "added_datetime",
        "window_size",
        "t_index",
        "label",
        "label_group",
    ]
    df = df.drop(columns=non_feature_cols)
    if drop_additional_features:
        df = df.drop(columns=drop_additional_features)

    # not sure what the issue is, but sklearn was throwing a future warning about the
    # feature names not all being strings, this fixes the issue...
    df.columns = [str(column) for column in df.columns]
    return df


def _get_registered_model_id(model_name: str, model_stage: str) -> str:
    """
    It gets the run_id of the registered model with the name and stage you specify

    Args:
      model_name (str): The name of the model you want to deploy.
      model_stage (str): The stage of the model to be used.

    Returns:
      The model_run_id
    """
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(f"http://{MLFLOW_DB_URI}")
    client = MlflowClient(f"http://{MLFLOW_DB_URI}")
    m_vs = client.search_model_versions(f"name='{model_name}'")
    for m_v in m_vs:
        mvx = dict(m_v)
        if mvx["current_stage"] == model_stage:
            model_run_id = mvx["run_id"]
            logger.info("model run_id = %s", model_run_id)
            break
    else:
        msg = "no %s model at %s stage", model_name, model_stage
        logger.error(msg)
        raise ValueError(msg)

    return model_run_id


def create_prediction_dataframe(  # pylint: disable=too-many-arguments
    feature_table: str,
    label_col: str,
    model_run_id: str,
    X_test: pd.DataFrame,
    y_test: Union[pd.core.series.Series, ndarray],
    y_pred: Union[pd.core.series.Series, ndarray],
) -> pd.DataFrame:
    """
    This function takes in the feature table name, the label column name, the model run id,
    the X_test dataframe, the y_test series, and the y_pred series, and returns a dataframe
    with the feature table id, the featurize id, the file name, the window size, the
    t_index, the true label, the predicted label, whether the prediction was correct, the
    model version, and the added datetime

    Args:
      feature_table (str): str,
      label_col (str): The column name of the label column in the feature table
      model_run_id (str): This is the unique identifier for the model run.
      X_test (pd.DataFrame): pd.DataFrame
      y_test (Union[pd.core.series.Series, ndarray]): the true labels
      y_pred (Union[pd.core.series.Series, ndarray]): the predicted labels

    Returns:
      A dataframe with the following columns:
        featurize_id
        file
        added_datetime
        window_size
        t_index
        true_label
        predicted_label
        prediction_correct
        model_version
    """

    pred_df = pd.DataFrame(
        columns=[
            "featurize_id",
            "file",
            "added_datetime",
            "window_size",
            "t_index",
            "true_label",
            "predicted_label",
            "prediction_correct",
            "model_version",
        ]
    )
    pred_correct = y_pred == y_test

    pred_df[feature_table + "_id"] = X_test[feature_table + "_id"]
    pred_df["featurize_id"] = X_test["featurize_id"]
    pred_df["file"] = X_test["file"]
    pred_df["window_size"] = X_test["window_size"]
    pred_df["t_index"] = X_test["t_index"]
    pred_df["true_label"] = X_test[label_col]
    pred_df["predicted_label"] = y_pred
    pred_df["prediction_correct"] = pred_correct
    pred_df["model_version"] = model_run_id
    pred_df["added_datetime"] = pd.to_datetime("now", utc=True)
    return pred_df


def write_predictions_to_db(
    df: pd.DataFrame, table: sa.sql.schema.Table, engine: sa.engine.base.Engine
) -> None:
    """
    It takes a dataframe, a table, and an engine, and writes the dataframe to the table
    using the engine

    Args:
      df (pd.DataFrame): the dataframe containing the features
      table (sa.sql.schema.Table): the table we're writing to
      engine (sa.engine.base.Engine): the database engine
    """
    records = df.to_dict(orient="records")
    with _session_scope(engine) as session:
        session.execute(table.insert(), records)
    sleep(2)


def _get_table_model(
    engine: sa.engine.base.Engine, metadata: sa.sql.schema.MetaData, table_name: str
) -> sa.sql.schema.Table:
    """
    It checks if a table exists in the database, and if it doesn't, it creates it. Finally,
    it returns a table object representation of the table in the database.

    Args:
      engine (sa.engine.base.Engine): the engine for the database connection
      metadata (sa.sql.schema.MetaData): the database metadata
      table_name (str): str = the name of the table to retrieve

    Returns:
      A table object
    """
    logger = logging.getLogger(__name__)
    try:
        table = sa.Table(table_name, metadata, autoload=True)
    except NoSuchTableError as err:
        logger.info(
            "table %s does not currently exist...attempting to create", table_name
        )
        table_obj = TABLE_DICT.get(table_name, None)
        if not table_obj:
            logger.error(
                (
                    "table %s not found in TABLE_DICT. table needs to be imported to "
                    "`predict_model_batch.py` from `predictions_tables.py` and added to "
                    "TABLE_DICT. may also require table definition in "
                    "`predictions_tables.py`",
                    table_name,
                )
            )
            raise NoSuchTableError from err

        table_obj.__table__.create(engine)
        sleep(2)
        logger.info("table created")
        table = sa.Table(table_name, metadata, autoload=True)

    return table


def apply_model(
    model_data: Dict[str, str],
    df_test_meta: pd.DataFrame,
    feature_table: str,
    label_col: str,
) -> pd.DataFrame:
    """
    This function takes in a model's name and stage, loads the model, and scores new data

    Args:
      model_data (Dict[str, str]): a dictionary containing the model name and stage
      df_test_meta (pd.DataFrame): the dataframe that contains the features and the label
      column
      feature_table (str): the name of the table that contains the features
      label_col (str): the name of the column in the feature table that contains the labels

    Returns:
      A dataframe with the predictions
    """
    logger = logging.getLogger(__name__)

    logger.info("performing preprocessing...")
    X_test = process_columns(feature_table, df_test_meta)
    records = X_test.to_dict(orient="records")
    y_test = df_test_meta[label_col].values
    logger.info("preprocessing complete")

    # load model
    logger.info("searching for model run_id...")
    model_run_id = _get_registered_model_id(model_data["name"], model_data["stage"])

    logger.info(
        ("loading model %s version of %s...", model_data["stage"], model_data["name"])
    )
    model_uri = f"runs:/{model_run_id}/"
    clf = mlflow.sklearn.load_model(model_uri)
    logger.info("loading complete")

    logger.info("scoring new data...")
    y_pred = clf.predict(records)
    acc = clf.score(records, y_test)
    logger.info("scoring complete...overall accuracy = %d", round(acc, 3))

    logger.info("generating predictions...")
    pred_df = create_prediction_dataframe(
        feature_table,
        label_col,
        model_run_id,
        df_test_meta,  # because X_test doesn't have id columns
        y_test,
        y_pred,
    )
    logger.info("predictions generated")
    return pred_df


@click.command()
@click.argument(
    "feature_table",
    type=str,
    required=False,
    default="naive_frequency_features",
)
@click.argument(
    "prediction_table",
    type=str,
    required=False,
    default="naive_frequency_features_predictions",
)
@click.argument(
    "label_col",
    type=str,
    required=False,
    default="label_group",
)
@click.argument(
    "model_name",
    type=str,
    required=False,
    default="exercise_prediction_naive_feats_pipe",
)
@click.argument(
    "model_stage",
    type=str,
    required=False,
    default="Production",
)
def main(
    feature_table: str = "naive_frequency_features",
    prediction_table: str = "naive_frequency_features_predictions",
    label_col: str = "label_group",
    model_name: str = "exercise_prediction_naive_feats_pipe",
    model_stage: str = "Production",
) -> None:
    """
    It loads simulated batch data from a table in the database, applies a model to it,
    and writes the predictions to a table in the database

    Args:
      feature_table (str): the name of the table in the feature store to load the
      data for scoring. Defaults to naive_frequency_features
      prediction_table (str): the name of the table in the feature store to log the
      predictions. Defaults to naive_frequency_features_predictions
      label_col (str): The name of the column in the feature table that contains the label.
      Defaults to label_group
      model_name (str): the name of the model in the model registry. Defaults to
      exercise_prediction_naive_feats_pipe
      model_stage (str): the stage of the model in the model registry. Defaults to Production
    """
    logger = logging.getLogger(__name__)

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
    table = _get_table_model(engine, metadata, prediction_table)
    write_predictions_to_db(pred_df, table, engine)
    logger.info("writing complete")
    logger.info("complete")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    FEATURE_STORE_URI = os.getenv("FEATURE_STORE_URI", "localhost")
    FEATURE_STORE_PW = os.getenv("FEATURE_STORE_PW")
    FEATURIZE_ID = os.getenv("FEATURIZE_ID")
    DATABASE_URI = (
        f"postgresql+psycopg2://postgres:{FEATURE_STORE_PW}@{FEATURE_STORE_URI}"
        "/feature_store"
    )

    MLFLOW_DB_URI = os.getenv("MLFLOW_DB_URI", "localhost:5000")
    # MLFLOW_DB_PW = os.getenv("MLFLOW_DB_PW")

    main()

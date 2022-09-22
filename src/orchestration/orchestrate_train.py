# -*- coding: utf-8 -*-
"""
This module defines the Prefect flow for orchestrating model training, using the
functions and methods defined in src.models.train_model.
"""
import os
from typing import Any, Dict, List, Optional, Union

import mlflow
from dotenv import find_dotenv, load_dotenv
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from prefect.task_runners import SequentialTaskRunner

import src.models.train_model as tm

ALL_SEARCH_PARAMS = tm.ALL_SEARCH_PARAMS

load_dotenv(find_dotenv())
FEATURE_STORE_URI = os.getenv("FEATURE_STORE_URI", "localhost:5432")
FEATURE_STORE_PW = os.getenv("FEATURE_STORE_PW")
DATABASE_URI = f"postgresql+psycopg2://postgres:{FEATURE_STORE_PW}@{FEATURE_STORE_URI}"
tm.EXP_NAME = os.getenv("EXP_NAME", "exercise_prediction_naive_feats")
DEBUG = os.getenv("DEBUG", "false") == "true"
if DEBUG:
    tm.EXP_NAME = tm.EXP_NAME + "_debug"

tm.DATABASE_URI = DATABASE_URI
tm.MLFLOW_TRACKING_SERVER = os.getenv("MLFLOW_TRACKING_SERVER", "localhost:5000")
tm.FEATURIZE_ID = os.getenv("FEATURIZE_ID")

mlflow.set_tracking_uri(f"http://{tm.MLFLOW_TRACKING_SERVER}")
mlflow.set_experiment(tm.EXP_NAME)

tm.CLIENT = MlflowClient(f"http://{tm.MLFLOW_TRACKING_SERVER}")
tm.EXP_ID = dict(mlflow.get_experiment_by_name(tm.EXP_NAME))["experiment_id"]

load_data = task(tm.load_data, name="Data Loading")  # type: ignore
process_columns = task(tm.process_columns, name="Preprocessing")  # type: ignore
model_search = task(  # type: ignore
    tm.model_search, name="Model Hyperparameter Search with hyperopt"
)
train_log_best_model = task(  # type: ignore
    tm.train_log_best_model, name="Train and Log Best Model"
)
test_log_best_model = task(  # type: ignore
    tm.test_log_best_model, name="Test and Log Best-Model Accuracy"
)
compare_with_registered_models = task(  # type: ignore
    tm.compare_with_registered_models, name="Compare with Registered Models"
)


@flow(
    name="Model Training",
    task_runner=SequentialTaskRunner(),
)
def train_flow(
    table_name: str = "naive_frequency_features",
    label_col: str = "label_group",
    model_search_json: str = "./model_search.json",
    initial_points_json: Optional[str] = None,
) -> None:
    # pylint: disable=too-many-locals
    # pylint: disable=protected-access
    """
    This function loads the data, performs a search over the hyperparameters using hyperopt,
    and then trains the best model on the training data and tests it on the test data.
    Finally, it compares the best model from this training to the existing model in
    production (if one exists) and registers and promotes the model to staging if either
    no production model exists or the new model has a better accuracy.

    Args:
      table_name (str): the name of the table in the database that contains the data
      label_col (str): the name of the column in the data that contains the labels
      model_search_json (str): This is the path to the JSON file that contains the model
      name, fixed parameters, and search parameters. Defaults to ./model_search.json
      initial_points_json (Optional[str]): This is the path to the JSON file that
      contains starting points for hyperparameter values for fitting procedure (e.g., to
      use values from previous fit to potentially speed up fitting). Defaults to None
    """
    logger = get_run_logger()
    logger.info("loading metadata")
    model_search_params = tm._read_json(model_search_json)
    data_limits = [
        model_search_params["train_limit"],
        model_search_params["validation_limit"],
    ]
    model_name = model_search_params["model"]
    fixed_params = model_search_params["fixed_paramaters"]
    search_params = model_search_params["search_parameters"]
    fmin_rstate = model_search_params["fmin_rstate"]

    initial_points: Optional[Union[List[Dict[Any, Any]], Dict[Any, Any]]] = None
    if initial_points_json:
        logger.info(
            "loading hyperparameter starting points from %s...", initial_points_json
        )
        initial_points = tm._read_json(initial_points_json)
        # if we have a dict of lists (i.e., multiple starting points), convert to list
        # of dicts
        if any(isinstance(val, list) for val in initial_points.values()):
            initial_points = [
                dict(zip(initial_points, t)) for t in zip(*initial_points.values())
            ]
        else:
            # assume otherwise it is a simply dict with 1 val per key
            initial_points = [initial_points]
        logger.info("trials will start with parameters %s", initial_points)

    logger.info("loading training and validation data...")
    df_train_meta, df_val_meta = (
        load_data(table_name, group, limit)
        for group, limit in zip(["train", "validation"], data_limits)
    )
    logger.info("loading complete")

    logger.info("performing preprocessing...")
    x_data = [process_columns(table_name, df) for df in (df_train_meta, df_val_meta)]
    y_data = [df_train_meta[label_col], df_val_meta[label_col]]
    logger.info("preprocessing complete")

    logger.info("performing model search for %s classifier...", model_name)
    search_space = {param: ALL_SEARCH_PARAMS[param] for param in search_params}
    best_params, parent_run_id = model_search(
        model_name,
        fixed_params,
        search_space,
        x_data,
        y_data,
        fmin_rstate,
        initial_points,
    )
    logger.info("model search complete...parent_run_id=%s", parent_run_id)
    logger.info("best parameters: %s", best_params)

    logger.info("logging best model in MLflow...")
    # first elements in x_data and y_data are training...
    best_clf, best_child_id = train_log_best_model(
        parent_run_id, model_name, best_params, x_data[0], y_data[0]
    )
    logger.info("logging complete...best_child_id=%s", best_child_id)

    logger.info("loading test data...")
    # test best model with test data
    df_test_meta = load_data(table_name, "test", model_search_params["test_limit"])
    logger.info("loading complete")

    logger.info("performing preprocessing...")
    X_test = process_columns(table_name, df_test_meta)
    y_test = df_test_meta[label_col]
    logger.info("preprocessing complete")

    logger.info("testing best model on test dataset...")
    best_acc = test_log_best_model(best_child_id, best_clf, X_test, y_test)
    logger.info("testing complete")

    logger.info("comparing best model with existing registered models")
    # compare test accuracy from best_child_id with accuracy from previous best
    # in best_run.json (if exists), update json if new accuracy better than previous
    _ = compare_with_registered_models(best_child_id, best_acc)
    logger.info("comparison complete")
    logger.info("complete")

# -*- coding: utf-8 -*-
"""
This module loads the data, performs a search over the hyperparameters using hyperopt,
and then trains the best model on the training data and tests it on the test data.
Finally, it compares the best model from this training to the existing model in production
(if one exists) and registers and promotes the model to staging if either no production
model exists or the new model has a better accuracy
"""
import gc
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import click
import mlflow
import pandas as pd
import sqlalchemy as sa
from dotenv import find_dotenv, load_dotenv
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from numpy import float64, random
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier  # NOQA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import sessionmaker

# there probably is a better way to map out the possible search-space criteria for
# different classifiers, but this will suffice for now to move the project along
ALL_SEARCH_PARAMS = {
    # randomforest, extratrees, and gradientboosting params
    "n_estimators": scope.int(hp.quniform("n_estimators", 100, 2000, 100)),
    "max_depth": scope.int(hp.quniform("max_depth", 4, 10, 1)),
    "min_samples_split-int": scope.int(hp.quniform("min_samples_split", 2, 6, 1)),
    "min_samples_leaf-int": scope.int(hp.quniform("min_samples_leaf", 1, 5, 1)),
    "min_samples_split": hp.loguniform("min_samples_split", -5, -2),
    "min_samples_split-float": hp.loguniform("min_samples_split", -5, -2),
    "min_samples_leaf": hp.loguniform("min_samples_leaf", -5, -2),
    "min_samples_leaf-float": hp.loguniform("min_samples_leaf", -5, -2),
    "min_weight_fraction_leaf": hp.quniform("min_weight_fraction_leaf", 0, 0.5, 0.05),
    "max_features": hp.choice("max_features", ("sqrt", "log2", None)),
    "max_leaf_nodes": hp.choice(
        "max_leaf_nodes", [None, hp.quniform("max_leaf_nodes-int", 10, 500, 10)]
    ),
    "min_impurity_decrease": hp.quniform("min_impurity_decrease", 0, 1, 0.05),
    "ccp_alpha": hp.loguniform("ccp_alpha", -4, 1),
    "max_samples-int": hp.choice(
        "max_samples", [None, hp.qloguniform("max_samples-int", 11, 13.5, 0.5)]
    ),
    # randomforest- and extratrees-specific params
    "max_samples": hp.quniform("max_samples", 0.5, 1, 0.05),
    "max_samples-float": hp.quniform("max_samples", 0.5, 1, 0.05),
    "class_weight": hp.choice("class_weight", ("balanced", "balanced_subsample", None)),
    # gradientboosting-specific params
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
}


def _read_json(file_path: Union[str, Path]) -> dict:
    """
    It reads a JSON file and returns the data as a dictionary

    Args:
      file_path (Union[str, Path]): The path to the JSON file.

    Returns:
      A dictionary
    """
    with open(file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data


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
    data_group: Literal["train", "validation", "test"],
    limit: Optional[int] = None,
):
    """
    It takes a table name, a data group, and an optional limit, and returns a pandas
    dataframe

    Args:
      table_name (str): The name of the table in the database that you want to load data
      from.
      data_group (Literal["train", "validation", "test"]): Which group of data you want
      load from the database. Options are "train" for training data, "validation" for
      validation data, or "test" for test data
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
):
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


def _get_named_classifier(model_name: str, params: Dict[str, Any]) -> ClassifierMixin:
    """
    It takes a classifier name and a dictionary of parameters, and returns a model object

    Args:
      model_name (str): name of the model
      params (Dict[str, Any]): Dict[str, Any] = parameters for the classifier

    Returns:
      A classifier object
    """
    if model_name.lower() not in (
        "randomforestclassifier",
        "gradientboostingclassifier",
        "extratreesclassifier",
    ):
        raise ValueError(f"{model_name} is not a recognized option")

    classifiers = {
        "extratreesclassifier": ExtraTreesClassifier,
        "randomforestclassifier": RandomForestClassifier,
        "gradientboostingclassifier": GradientBoostingClassifier,
    }

    classifier = classifiers[model_name.lower()]
    clf = classifier(**params)
    d_v = DictVectorizer()
    pipe = Pipeline([("dv", d_v), ("clf", clf)])
    return pipe


def model_search(
    model_name: str,
    fixed_params: Dict[str, Any],
    search_space: Dict[str, Any],
    x_data: List[pd.DataFrame],
    y_data: List[pd.core.series.Series],
    rstate: Optional[Union[int, float]] = None,
    initial_points: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], str]:
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-arguments
    """
    Performs a model search of hyperparameters using hyperopt given the model name,
    the search space, and training/validation data and returns the parameters from the
    best run and the MLflow parent run_id.

    Args:
      model_name (str): str, the name of the classifier to use
      fixed_params (Dict[str, Any]): These are fixed parameters that will be the same for
      every run of the model search.
      search_space (Dict[str, Any]): A dictionary of the search space for each hyperparameter,
      in the format required by hyperopt.
      x_data (List[pd.DataFrame]): feature data for training and
      validation, in the form of (X_train, X_val)
      y_data (List[pd.core.series.Series]): label data for training
      and validation, in the form of (y_train, y_val)
      initial_points (Optional[List[Dict[str, Any]]]): dictionary of initial points to
      use as start of hyperparameter search. defaults to None

    Returns:
      The best result and the parent run id.
    """
    X_train, X_val = x_data
    y_train, y_val = y_data

    def _objective(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        It takes a dictionary of hyperparameters, creates a new MLflow run, logs the
        hyperparameters, fits a model, logs the accuracy, and returns a dictionary with the
        loss and status

        Args:
          params (Dict[str, Any]): dictionary of parameters for the classifier

        Returns:
          The loss and status.
        """
        with mlflow.start_run(run_name="standalone-hyperopt-fit-child", nested=True):
            mlflow.set_tags(
                {
                    "developer": "adam gifford",
                    "model": model_name,
                    "N_train": X_train.shape[0],
                    "N_validation": X_val.shape[0],
                    "fmin_rstate": rstate,
                }
            )

            params.update(fixed_params)
            clf = _get_named_classifier(model_name, params)
            mlflow.log_params(params)

            clf.fit(X_train.to_dict(orient="records"), y_train.values)
            acc = clf.score(X_val.to_dict(orient="records"), y_val.values)
            mlflow.log_metric("validation_accuracy", acc)

            del clf
            gc.collect()

        return {"loss": -acc, "status": STATUS_OK}

    d_t = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    with mlflow.start_run(run_name="standalone-hyperopt-fit " + d_t) as run:
        parent_run_id = run.info.run_id

        if initial_points:
            trials = generate_trials_to_calculate(initial_points)
        else:
            trials = Trials()

        best_result = fmin(
            fn=_objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
            rstate=random.default_rng(rstate),
            early_stop_fn=no_progress_loss(15),
        )

    best_result = space_eval(search_space, best_result)
    best_result.update(fixed_params)
    return best_result, parent_run_id


def train_log_best_model(
    parent_run_id: str,
    model_name: str,
    best_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.core.series.Series,
) -> Tuple[ClassifierMixin, str]:
    """
    It takes the best hyperparameters from the best child run in the parent run, trains a
    model with those hyperparameters, and logs the model back to the best child run

    Args:
      parent_run_id (str): The run ID of the parent run.
      model_name (str): The name of the model you want to train.
      best_params (Dict[str, Any]): The best parameters found by the hyperparameter tuning
      run.
      X_train (pd.DataFrame): The training data
      y_train (pd.core.series.Series): The target variable

    Returns:
      The fitted classifier and best child run id
    """

    clf = _get_named_classifier(model_name, best_params)

    # get run_id from best run of parent_run_id based on acc DESC
    best_child_run = CLIENT.search_runs(
        experiment_ids=EXP_ID,
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.validation_accuracy DESC"],
    )[0]

    best_child_id = best_child_run.info.run_id
    with mlflow.start_run(run_id=best_child_id):
        clf.fit(X_train.to_dict(orient="records"), y_train.values)
        mlflow.sklearn.log_model(clf, artifact_path="artifacts")

    return clf, best_child_id


def test_log_best_model(
    run_id: str,
    clf: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.core.series.Series,
) -> float:
    """
    It takes a run ID, a classifier, and a test set, and logs the accuracy of the classifier
    on the test set

    Args:
      run_id (str): The run ID of the run that produced the best model.
      clf (ClassifierMixin): The model we want to score and log metric for
      X_test (pd.DataFrame): The test data
      y_test (pd.core.series.Series): The actual labels of the test data
    """
    with mlflow.start_run(run_id=run_id):
        acc = clf.score(X_test.to_dict(orient="records"), y_test.values)
        mlflow.log_metric("test_accuracy", acc)

    return acc


def register_model(run_id: str) -> str:
    """
    It takes a run ID as input, registers the model for that run, and returns the model
    version of the registered model.

    Args:
      run_id (str): The run ID of the run that produced the model to register.

    Returns:
      The model version
    """
    model_uri = f"runs:/{run_id}/models"
    mdl_ver_metadata = mlflow.register_model(model_uri=model_uri, name=EXP_NAME)
    return mdl_ver_metadata.version


def transition_model_and_log(
    model_version: str, model_stage: str, archive_existing_versions: bool = False
):
    """
    It transitions a model version to a new stage and logs the transition in the model
    version's description

    Args:
      model_version (str): The version of the model you want to transition.
      model_stage (str): The stage to transition the model version to.
      archive_existing_versions (bool): If True, the existing versions of the model will be
      archived. Defaults to False
    """
    CLIENT.transition_model_version_stage(
        name=EXP_NAME,
        version=model_version,
        stage=model_stage,
        archive_existing_versions=archive_existing_versions,
    )

    date = datetime.today()
    CLIENT.update_model_version(
        name=EXP_NAME,
        version=model_version,
        description=(
            f"The model version {model_version} was transitioned to {model_stage} on "
            f"{date}"
        ),
    )
    return "complete"


def compare_with_registered_models(
    new_model_run_id: str, new_acc: Union[float64, float]
) -> str:
    """
    If the new model is better than existing registered models, register the new model
    and promote it to staging

    Args:
      new_model_run_id (str): The run ID of the new model.
      new_acc (Union[float64, float]): The accuracy of the new model.

    Returns:
      The model version
    """
    logger = logging.getLogger(__name__)
    # first, need to check if model exists in the registry, create if not
    registered_models = CLIENT.list_registered_models()
    model_names = []
    for r_m in registered_models:
        r_m = dict(r_m)
        model_names.append(r_m["name"])

    if EXP_NAME not in model_names:
        logger.info("model %s does not exist, creating empty model registry...")
        CLIENT.create_registered_model(EXP_NAME)

    # get production model run_id
    prod_version = CLIENT.get_latest_versions(name=EXP_NAME, stages=["Production"])
    stage_version = CLIENT.get_latest_versions(name=EXP_NAME, stages=["Staging"])

    # if no registered models exist, register automatically and promote to staging
    if not prod_version and not stage_version:
        logger.info(
            (
                "no previous versions exist in model registry, registering model and "
                "promoting to Staging..."
            )
        )
        model_version = register_model(new_model_run_id)
        return transition_model_and_log(model_version, "Staging", True)

    if not prod_version:  # else if only staging exists, compare with staging
        logger.info(
            (
                "no Production version exists in model registry, comparing to existing "
                "model in Staging..."
            )
        )
        prev_run_id = stage_version[0].run_id
        prev_acc = CLIENT.get_run(run_id=prev_run_id).data.metrics["test_accuracy"]
    elif not stage_version:  # else if only prod exists, compare with prod
        logger.info(
            (
                "only Production version exists in model registry, comparing to existing "
                "model in Production..."
            )
        )
        prev_run_id = prod_version[0].run_id
        prev_acc = CLIENT.get_run(run_id=prev_run_id).data.metrics["test_accuracy"]
    else:  # both exist, compare with the best of the two
        logger.info(
            (
                "models exist in both Production and Staging, identifying best model to "
                "compare to..."
            )
        )
        prod_run_id = prod_version[0].run_id
        prod_acc = CLIENT.get_run(run_id=prod_run_id).data.metrics["test_accuracy"]
        stage_run_id = stage_version[0].run_id
        stage_acc = CLIENT.get_run(run_id=stage_run_id).data.metrics["test_accuracy"]
        prev_acc = prod_acc if prod_acc >= stage_acc else stage_acc

    # if new better than old, register new and promote to staging (promotion to prod
    # will be manual for the sake of this project)
    if new_acc > prev_acc:
        logger.info(
            (
                "new model's accuracy is better than existing registered models, promoting"
                " new model to Staging..."
            )
        )
        model_version = register_model(new_model_run_id)
        return transition_model_and_log(model_version, "Staging", True)

    logger.info(
        (
            "new model's accuracy is worse than existing registered models, not promoting"
            " new model to the registry..."
        )
    )
    return "complete"


@click.command()
@click.argument(
    "table_name",
    type=str,
    required=False,
    default="naive_frequency_features",
)
@click.argument(
    "label_col",
    type=str,
    required=False,
    default="label_group",
)
@click.argument(
    "model_search_json",
    type=click.Path(exists=True),
    required=False,
    default="./model_search.json",
)
@click.option(
    "--initial_points_json",
    type=click.Path(exists=False),
    required=False,
    default=None,
    show_default=True,
    help=(
        (
            "JSON file containing hyperparameter starting points for first trial in "
            "hyperopt fit."
        )
    ),
)
def main(
    table_name: str = "naive_frequency_features",
    label_col: str = "label_group",
    model_search_json: str = "./model_search.json",
    initial_points_json: Optional[str] = None,
) -> None:
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
    # pylint: disable=too-many-locals
    logger = logging.getLogger(__name__)
    logger.info("loading metadata")
    model_search_params = _read_json(model_search_json)
    data_limits = [
        model_search_params["train_limit"],
        model_search_params["validation_limit"],
    ]
    model_name = model_search_params["model"]
    fixed_params = model_search_params["fixed_paramaters"]
    search_params = model_search_params["search_parameters"]
    fmin_rstate = model_search_params["fmin_rstate"]

    initial_points = None
    if initial_points_json:
        logger.info(
            "loading hyperparameter starting points from %s...", initial_points_json
        )
        initial_points_ = _read_json(initial_points_json)
        # if we have a dict of lists (i.e., multiple starting points), convert to list
        # of dicts
        if any(isinstance(val, list) for val in initial_points_.values()):
            initial_points = [
                dict(zip(initial_points_, t)) for t in zip(*initial_points_.values())
            ]
        else:
            # assume otherwise it is a simply dict with 1 val per key
            initial_points = [initial_points_]
        logger.info("trials will start with parameters %s", initial_points)

    logger.info("loading training and validation data...")
    df_train_meta, df_val_meta = (
        load_data(table_name, group, limit)  # type: ignore
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


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    FEATURE_STORE_URI = os.getenv("FEATURE_STORE_URI", "localhost:5432")
    FEATURE_STORE_PW = os.getenv("FEATURE_STORE_PW")
    FEATURIZE_ID = os.getenv(
        "FEATURIZE_ID",
        "b924133661c11af1c2c7f560527c03833cec4dfd2202b30afe058b5d61d176e7",
    )
    EXP_NAME = os.getenv("EXP_NAME", "exercise_prediction_naive_feats")
    DEBUG = os.getenv("DEBUG", "false") == "true"
    DATABASE_URI = (
        f"postgresql+psycopg2://postgres:{FEATURE_STORE_PW}@{FEATURE_STORE_URI}"
        "/feature_store"
    )

    MLFLOW_DB_URI = os.getenv("MLFLOW_DB_URI", "localhost:5000")
    # MLFLOW_DB_PW = os.getenv("MLFLOW_DB_PW")

    mlflow.set_tracking_uri(f"http://{MLFLOW_DB_URI}")
    if DEBUG:
        EXP_NAME = EXP_NAME + "_debug"

    mlflow.set_experiment(EXP_NAME)
    EXP_ID = dict(mlflow.get_experiment_by_name(EXP_NAME))["experiment_id"]
    CLIENT = MlflowClient(f"http://{MLFLOW_DB_URI}")

    main()

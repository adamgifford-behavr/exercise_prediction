# -*- coding: utf-8 -*-
"""
This module defines the functions and methods that create the training, validation, and
testing datasets from the raw PARQUET files. The `main()` function takes

Args:

"""
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from time import sleep
from typing import Any, Dict, Generator, Iterable, Tuple, Union

import click
import numpy as np
import numpy.matlib
import pandas as pd
import sqlalchemy as sa
from dotenv import find_dotenv, load_dotenv
from scipy import signal
from scipy.fftpack import fft, fftshift
from sqlalchemy.orm import sessionmaker


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


def _calculate_frequencies(n_fft: int, fs: int = 50) -> np.ndarray:
    """
    It calculates the frequency vector for a given FFT length and sampling frequency

    Args:
      n_fft (int): The number of points in the FFT.
      fs (int): sampling frequency. Defaults to 50

    Returns:
      The frequencies of the FFT.
    """
    n_points = 2 * int(np.floor(n_fft / 2))
    if n_fft % 2:
        n_points += 1
    freq = fs / 2 * np.linspace(-1, 1, n_points)
    return freq


def _calculate_spectrum(ndarray: np.ndarray) -> np.ndarray:
    """
    It takes a 2D numpy array, performs a 1D FFT on each row, and then normalizes the result

    Args:
      ndarray (np.ndarray): the input data

    Returns:
      The return value is a numpy array of the same shape as the input array.
    """
    X_w = fftshift(fft(ndarray, axis=0), axes=0)
    return np.abs(X_w / numpy.matlib.repmat(abs(X_w).max(axis=0), ndarray.shape[0], 1))


def _get_frequencies_indices(
    all_freqs: np.ndarray, desired_freqs: Iterable
) -> np.ndarray:
    """
    It takes a list of all frequencies and a list of desired frequencies, and returns the
    indices of the closest frequencies in the list of all frequencies

    Args:
      all_freqs (np.ndarray): all frequencies of the FFT to be computed
      desired_freqs (Iterable): the frequencies you want to extract from the data

    Returns:
      The indices of the closest frequencies in the all_freqs array to the desired_freqs
    array.
    """
    closest_freqs_ix = np.array(
        [(np.abs([af - df for af in all_freqs])).argmin() for df in desired_freqs]
    )
    return closest_freqs_ix


def _get_labels(df_snip: pd.DataFrame) -> Tuple[str, str]:
    """
    It takes a dataframe of a single time period and returns the label and label group
    of that time period

    Args:
      df_snip (pd.DataFrame): the dataframe of the current time period

    Returns:
      A tuple of the label and label_group
    """
    # create new label in case the time period encapsulates parts of multiple activity groups
    if df_snip.label.nunique() > 1:
        label = "Transition"
    else:
        label = df_snip.loc[0, "label"]

    if df_snip.label_group.nunique() > 1:
        label_group = "Transition"
    else:
        label_group = df_snip.loc[0, "label_group"]

    return label, label_group


def _calculate_single_window_features(X_w: np.ndarray, freq_ixs: Iterable) -> list:
    """
    It takes a windowed data spectrum across raw measurements and a list of frequency
    indices, and returns a flattened list of features for each measurement and desired
    frequency in the window

    Args:
      X_w (np.ndarray): the windowed data in the shape of n_frequencies x n_columns
      freq_ixs (Iterable): the indices of the frequencies we want to use for each
      measurement

    Returns:
      A list of features for each window.
    """
    # get only the desired frequencies by measurement
    meas_feats = [X_w[f_ix, ix] for ix, f_ix in enumerate(freq_ixs)]
    # flatten the features to have same shape as feat_cols
    flat_feats = [f_data for col in meas_feats for f_data in col]
    return flat_feats


def _make_single_window_df(
    featurize_id: str,
    df_file: str,
    dataset_group: str,
    n_fft: int,
    ix: int,
    label: str,
    label_group: str,
    feat_cols: list,
    flat_feats: list,
) -> pd.DataFrame:
    """
    This function takes in a set of arguments defining the metaparameters and features
    for a single window of time in the dataset and returns 1-row dataframe

    Args:
      featurize_id (str): a unique identifier for the featurization job
      df_file (str): the name of the file that the window is from
      dataset_group (str): the name of the dataset group
      n_fft (int): The number of samples in each window.
      ix (int): the index of the window in the file
      label (str): the label of the file
      label_group (str): the group that the label belongs to
      feat_cols (list): list of feature names
      flat_feats (list): list of features for a single window

    Returns:
      A dataframe with the following columns:
        featurize_id,
        file,
        dataset_group,
        added_datetime,
        window_size,
        t_index,
        label,
        label_group,
        feat_cols,
        flat_feats
    """
    # pylint: disable=too-many-arguments
    data = {
        "featurize_id": [featurize_id],
        "file": [df_file],
        "dataset_group": [dataset_group],
        "added_datetime": [pd.to_datetime("now", utc=True)],
        "window_size": [n_fft],
        "t_index": [ix],
        "label": [label],
        "label_group": [label_group],
        **{key: [val] for key, val in zip(feat_cols, flat_feats)},
    }
    window_df = pd.DataFrame(data=data)
    return window_df


# def _generate_window_fxn(window_name: str, n_fft: int, n_cols: int) -> np.ndarray:
#     """
#     This function generates a window function of a given type and size

#     Args:
#       window_name (str): str
#       n_fft (int): The number of samples in each STFT window.
#       n_cols (int): number of columns in the spectrogram

#     Returns:
#       A window function.
#     """
#     if "hann" in window_name.lower():
#         window = numpy.matlib.repmat(signal.hann(n_fft), n_cols, 1).T
#     else:
#         raise ValueError(f"window_name {window_name} not instantiated")
#     return window


def _make_empty_features_df(table: sa.sql.schema.Table) -> pd.DataFrame:
    """
    It takes a SQLAlchemy table object and returns a Pandas dataframe with the same columns

    Args:
      table (sa.sql.schema.Table): The table object that we want to create a dataframe for.

    Returns:
      A dataframe with the columns of the table.
    """
    columns = [column.name for column in table.columns]
    # remove primary key col since it will be autogenerated
    df = pd.DataFrame(columns=columns).drop(columns=[table.fullname + "_id"])
    return df


def calculate_windowed_feats(
    featurize_id: str,
    dataset_group: str,
    df_file: str,
    n_fft: int,
    features: Dict[str, list],
    table_columns: Iterable,
) -> pd.DataFrame:
    """
    This function takes in a set of metaparameters and a dictionary of features to
    calculate, and returns a dataframe of the calculated features by time window

    Args:
      featurize_id (str): a unique id for the featurization process
      dataset_group (str): the name of the dataset group (i.e. "train", "test", "validation")
      df_file (str): the path to the parquet file containing the data
      n_fft (int): the number of samples to use in the FFT
      features (Dict[str, list]): a dictionary of the form {'measurement_col_name':
      [freq1, freq2, ...]}
      table_columns (Iterable): the columns of the table that will be created in the database

    Returns:
      A dataframe with the features for each window.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    df = pd.read_parquet(df_file, engine="fastparquet")
    measurement_cols = list(features.keys())

    # window = _generate_window_fxn(metaparams["window"], n_fft, len(measurement_cols))
    window = numpy.matlib.repmat(signal.hann(n_fft), len(measurement_cols), 1).T

    freq = _calculate_frequencies(n_fft)
    freq_ixs = [
        _get_frequencies_indices(freq, features[col]) for col in measurement_cols
    ]
    feat_cols = [
        col + "_" + str(int(freq_feat))
        for col in measurement_cols
        for freq_feat in features[col]
    ]

    features_df = pd.DataFrame(columns=table_columns)
    for ix, t_start in enumerate(range(0, df.shape[0], n_fft)):
        if t_start + n_fft > df.shape[0] - 1:
            continue

        df_snip = df.loc[t_start : (t_start + n_fft - 1), :].reset_index()
        label, label_group = _get_labels(df_snip)

        # compute the flattened_features and make the windowed feature df
        X_w = _calculate_spectrum(window * df_snip[measurement_cols].values)
        flat_feats = _calculate_single_window_features(X_w, freq_ixs)
        window_df = _make_single_window_df(
            featurize_id,
            df_file,
            dataset_group,
            n_fft,
            ix,
            label,
            label_group,
            feat_cols,
            flat_feats,
        )

        features_df = pd.concat([features_df, window_df], ignore_index=True)

    return features_df


def _generate_featurize_id(
    train_val_files: Dict[str, list], metaparams: Dict[str, str]
) -> str:
    """
    Given a dictionary of file paths and a dictionary of metaparameters, generate a
    unique ID for the featurization process

    Args:
      train_val_files (Dict[str, list]): a dictionary with keys "train" and "validation"
      and values that are lists of file paths
      metaparams (Dict[str, str]): the additional metaparameters that uniquely identify
      the featurization process

    Returns:
      A dictionary of the train, validation, and test files.
    """

    # assumption is all files are used for train, val, or test, so we only need to
    # add 2 of the 3 in order to infer the third
    # optionally, can add a new line for test file list, but need to add input to fxn
    # for that
    metadata = copy(train_val_files["train"])
    metadata.extend(train_val_files["validation"])
    # the rest of the uniquely identifying parameters (e.g., table_name, nfft,
    # featurization pipeline, etc.), are found in the metaparameters
    metadata.extend(["".join(key + str(val) for key, val in metaparams.items())])

    return hashlib.sha256("".join(metadata).encode()).hexdigest()


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
    except:
        session.rollback()
        raise
    finally:
        session.close()


def write_features_to_db(
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


def check_for_existing_id(
    featurize_id: str, table: sa.sql.schema.Table, engine: sa.engine.base.Engine
) -> bool:
    """
    It checks if a featurize_id already exists in the database

    Args:
      featurize_id (str): str
      table (sa.sql.schema.Table): the table to check for the featurize_id
      engine (sa.engine.base.Engine): the SQLAlchemy engine to use to connect to the database

    Returns:
      A boolean value.
    """
    with _session_scope(engine) as session:
        featurize_ids = session.query(table.c.featurize_id).distinct().all()

    return any(row[0] == featurize_id for row in featurize_ids)


def delete_existing_dataset(
    featurize_id: str, table: sa.sql.schema.Table, engine: sa.engine.base.Engine
) -> None:
    """
    It deletes all rows in the table that have the same featurize_id as the one we're
    about to insert

    Args:
      featurize_id (str): This is the unique identifier for the dataset.
      table (sa.sql.schema.Table): The table that we want to delete the data from.
      engine (sa.engine.base.Engine): the database engine
    """
    with _session_scope(engine) as session:
        session.query(table).filter(table.c.featurize_id == featurize_id).delete()
    sleep(2)


@click.command()
@click.argument(
    "features_json",
    type=click.Path(exists=True),
    required=False,
    default="./frequency_features.json",
)
@click.argument(
    "data_splits_json",
    type=click.Path(exists=True),
    required=False,
    default="./datafile_group_splits.json",
)
@click.argument(
    "metaparams_json",
    type=click.Path(exists=True),
    required=False,
    default="./metaparams.json",
)
@click.option(
    "--overwrite-data/--keep-data",
    type=bool,
    default=False,
    show_default=True,
    help=(
        "Whether to overwrite data of matching metaparameters (determined via matching "
        "featurize_id) if they already exist."
    ),
)
def main(
    features_json: str = "./frequency_features.json",
    data_splits_json: str = "./datafile_group_splits.json",
    metaparams_json: str = "./metaparams.json",
    overwrite_data: bool = False,
):
    """
    It takes in a set of data file paths, calculates features for each file, and writes
    the features to a database, organized by dataset group (i.e., train, val, test)

    Args:
      features_json (str): The path to the json file that lists the frequencies of
      interest for each measurement signal. Defaults to ./frequency_features.json
      data_splits_json (str): The path to the json file that separates the files into
      training, validation, testing, and simulation groupings. Defaults to
      ./datafile_group_splits.json
      metaparams_json (str): str = The path to the json file that describes the
      metaparameters that uniquely identify the processing pipeline to build the
      features dataset. Defaults to ./metaparams.json
      overwrite_data (bool): Whether to overwrite an existing dataset based on matching
      featurize_id, if it exists. Defaults to False
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    logger = logging.getLogger(__name__)

    # load meta data
    logger.info("loading metadata")
    datafile_splits = _read_json(data_splits_json)
    features = _read_json(features_json)
    metaparams = _read_json(metaparams_json)

    table_name = metaparams["table_name"]
    n_fft = metaparams["n_fft"]

    featurize_id = _generate_featurize_id(datafile_splits, metaparams)
    logger.info("id for this run of `build_features` is %s", featurize_id)

    engine: sa.engine.base.Engine = sa.create_engine(
        DATABASE_URI,
        executemany_mode="values",
        executemany_values_page_size=10000,
        executemany_batch_page_size=500,
    )
    metadata = sa.schema.MetaData(bind=engine)
    table = sa.Table(table_name, metadata, autoload=True)

    prev_feat_id_exists = check_for_existing_id(featurize_id, table, engine)
    if prev_feat_id_exists and overwrite_data:
        logger.info(
            (
                "previous features dataset exists but overwrite_data=True. deleting "
                "existing dataset..."
            )
        )
        delete_existing_dataset(featurize_id, table, engine)
        logger.info("dataset deleted")

    if (not prev_feat_id_exists) or overwrite_data:
        logger.info("making features dataset...")

        all_features_df = _make_empty_features_df(table)
        for dataset_group, data_files in datafile_splits.items():
            logger.info("creating %s dataset...", dataset_group)
            for file in data_files:
                logger.info("analyzing file %s...", file)
                features_df = calculate_windowed_feats(
                    featurize_id,
                    dataset_group,
                    file,
                    n_fft,
                    features,
                    all_features_df.columns,
                )
                all_features_df = pd.concat([all_features_df, features_df])

            logger.info("%s dataset complete", dataset_group)

        # logger.info("creating test dataset")
        # test_files = trainval_test_files["test"]
        # for file in test_files:
        #     logger.info("analyzing file %s...", file)
        #     features_df = calculate_windowed_feats(
        #         featurize_id, "test", file, n_fft, features, all_features_df.columns
        #     )
        #     all_features_df = pd.concat([all_features_df, features_df])

        # logger.info("test dataset complete")

        logger.info("writing dataset to database")
        write_features_to_db(all_features_df, table, engine)
        logger.info("writing to database complete")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[1]
    FEATURE_STORE_URI = os.getenv("FEATURE_STORE_URI", "localhost")
    FEATURE_STORE_PW = os.getenv("FEATURE_STORE_PW")
    DATABASE_URI = (
        f"postgresql+psycopg2://postgres:{FEATURE_STORE_PW}@{FEATURE_STORE_URI}"
        "/feature_store"
    )

    main()

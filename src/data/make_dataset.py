# -*- coding: utf-8 -*-
"""
This module defines the functions and methods that create the datasets from the raw data
files. The `main()` function takes two arguments, input_filepath and output_filepath, and
...

Args:
    input_filepath: The path to the input file.
    output_filepath: The path to the output file.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Union

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from scipy import io as sio

ACTIVITY_GROUPS_FILE = Path(__file__).parent / "activity_groupings.json"


def _load_matfile(file: str) -> dict:
    """
    It loads a .mat file and returns a dictionary

    Args:
      file (str): fulle file path of the .mat file

    Returns:
      A dictionary of the contents of the .mat file.
    """
    return sio.loadmat(file, squeeze_me=True, struct_as_record=False)


def _align_activities(
    activity_start_matrix: np.ndarray, time: np.ndarray
) -> np.ndarray:
    """
    It takes a matrix of activity names and start/end times, and returns an array of
    activity names that aligns with the time array

    Args:
      activity_start_matrix (np.ndarray): a numpy array of shape (n_activities, 3) where the
      first column is the activity name, the second column is the start time, and the third
      column is the end time.
      time (np.ndarray): the time vector

    Returns:
      An array of activity names aligned to time.
    """
    max_t = time[-1]
    activity_array = np.empty_like(time, dtype="object")
    for activity in activity_start_matrix:
        # data in form of [activity_name, start_time, end_time]
        activity_name = activity[0]
        t_s, t_e = activity[1:3]

        # enforce that times fit between start and end times in data matrix
        t_s = 0 if t_s < 0 else t_s
        t_e = max_t if t_e > max_t else t_e

        act_ix = (time >= t_s) & (time <= t_e)
        activity_array[act_ix] = activity_name

    return activity_array


def _reverse_dict(activity_groupings: dict) -> dict:
    """
    It takes a dictionary of activity groupings and returns a dictionary of activities to
    groupings

    Args:
      activity_groupings (dict): a dictionary of activity groupings. The keys are the names
      of the groups, and the values are lists of activity indices

    Returns:
      A dictionary with the activities as the keys and the activity groups as the values.
    """
    groupings_activities = {
        val_ix: key for key, val in activity_groupings.items() for val_ix in val
    }
    return groupings_activities


def _write_single_parquet_file(
    interim_filepath: Union[str, Path],
    subj_data: sio.matlab.mat_struct,
    data_id: int,
) -> None:
    """
    It takes a matlab struct loaded with scipy, converts it to a pandas dataframe, and
    writes it to a parquet file

    Args:
      interim_filepath (str): full interim file path to save the parquet file
      subj_data (sio.matlab.mio5_params.mat_struct): subject's mat_struct data
      data_id (int): data index for subject (0 by default if only one data vector for a user)
    """

    time = subj_data.data.accelDataMatrix[:, 0]
    file_id = subj_data.fileIndex
    subject_id = subj_data.subjectID

    df = pd.DataFrame()
    df["time"] = time
    df["file_id"] = file_id
    df["subject_id"] = subject_id
    df["data_id"] = data_id

    df["accel_x"] = subj_data.data.accelDataMatrix[:, 1]
    df["accel_y"] = subj_data.data.accelDataMatrix[:, 2]
    df["accel_z"] = subj_data.data.accelDataMatrix[:, 3]

    df["gyro_x"] = subj_data.data.gyroDataMatrix[:, 1]
    df["gyro_y"] = subj_data.data.gyroDataMatrix[:, 2]
    df["gyro_z"] = subj_data.data.gyroDataMatrix[:, 3]

    activity_array = _align_activities(subj_data.activityStartMatrix, time)
    df["label"] = activity_array

    with open(ACTIVITY_GROUPS_FILE, "r", encoding="utf-8") as infile:
        activity_groupings = json.load(infile)

    # want to reverse relationship between keys and values for easy lookup by activity
    groupings_actvitiy = _reverse_dict(activity_groupings)
    df["label_group"] = df["label"].apply(
        lambda activity: groupings_actvitiy.get(activity, None)
    )

    df.to_parquet(interim_filepath, engine="fastparquet")


def _write_activity_groupings_json(useful_activity_groupings: np.ndarray) -> None:
    """
    It takes the useful activity groupings data and writes them to a JSON file

    Args:
    usefulActivityGroupings (np.ndarray): a 2D numpy array of the form:
        `array([group_name, array([labels for this group]]))`
    """
    activity_groupings = {row[0]: row[1].tolist() for row in useful_activity_groupings}

    with open(ACTIVITY_GROUPS_FILE, "w", encoding="utf-8") as outfile:
        json.dump(activity_groupings, outfile)


def write_single_parquet_file_wrapper(
    interim_path: Union[str, Path],
    subj_data: np.ndarray,
    data_id: Optional[int] = 0,
    overwrite: Optional[bool] = False,
):
    """
    Wrapper function that takes an array of subject data, calls `_write_single_parquet_file`
    to write it to a parquet file, and returns nothing

    Args:
      interim_path (str): full interim file path to save the parquet file
      subj_data (np.ndarray): array of subject data
      data_id (Optional[int]): Optional[int] = 0,. Defaults to 0
      overwrite (Optional[bool]): bool = False,. Defaults to False
    """
    logger = logging.getLogger(__name__)

    file_id = subj_data.fileIndex
    subject_id = subj_data.subjectID

    interim_filepath = (
        Path(interim_path)
        / f"fileID{file_id}_subjID{subject_id}_dataID{data_id}.parquet"
    )
    if (not interim_filepath.exists()) or overwrite:
        logger.info("writing file %s...", interim_filepath)
        _write_single_parquet_file(interim_filepath, subj_data, data_id)
        logger.info("file %s complete", interim_filepath)
    else:
        logger.info("file %s already exists, skipping...", interim_filepath)


def multi_convert_mat_to_parquet(
    input_filepath: str, interim_path: str, overwrite
) -> None:
    """
    It takes a .mat file path, loads it, and then iterates through the contents of the file,
    writing each subject's data to separate parquet files

    Args:
      input_filepath (str): the path to the .mat file
      interim_path (str): the path to the interim folder
      overwrite: boolean, whether to overwrite existing files
    """
    logger = logging.getLogger(__name__)

    mat_contents = _load_matfile(input_filepath)
    if not ACTIVITY_GROUPS_FILE.exists():
        logger.info("activity groupings file does not exist. writing file first...")
        _write_activity_groupings_json(
            mat_contents["exerciseConstants"].usefulActivityGroupings
        )

    # loop over contents to re-write the data by file_id, subject_id, and data_id
    for subj_data in mat_contents["subject_data"]:
        if isinstance(subj_data, np.ndarray):
            for d_ix, subj_data_x in enumerate(subj_data):
                write_single_parquet_file_wrapper(
                    interim_path, subj_data_x, d_ix, overwrite
                )
        else:
            write_single_parquet_file_wrapper(interim_path, subj_data, 0, overwrite)


@click.command()
@click.argument(
    "input_filepath",
    type=click.Path(exists=True),
    required=False,
    default=None,
)
@click.argument(
    "interim_path",
    type=click.Path(),
    required=False,
    default=None,
)
@click.argument(
    "output_path",
    type=click.Path(),
    required=False,
    default=None,
)
@click.option(
    "--overwrite_interim/--keep_interim",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to overwrite files of matching interim data by filename if they already exist.",
)
@click.option(
    "--overwrite_output/--keep_output",
    type=bool,
    default=False,
    show_default=True,
    help="Whether to overwrite files of matching output data by filename if they already exist.",
)
def main(
    input_filepath: Optional[str] = None,
    interim_path: Optional[str] = None,
    output_path: Optional[str] = None,
    overwrite_interim: Optional[bool] = False,
    overwrite_output: Optional[bool] = False,
) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../interim/preprocessed). It requires an
    intermediate step of converting data in a .mat file to a series of PARQUET files (one
    file per subject and run through an exercise routine)

    Args:
      input_filepath (Optional[str]): location of the raw .mat file. Defaults to None,
      which is replaced by "../../data/raw/exercise_data.50.0000_multionly.mat" in the
      function
      interim_path (Optional[str]): location to store the PARQUET files. Defaults to
      None, which is replaced by "../../data/interim/raw/" in the function
      output_path (Optional[str]): location to store the final preprocessed output.
      Defaults to None, which is replaced by "../../data/interim/preprocessed/" in the
      function
      overwrite_interim (Optional[bool]): Whether to overwrite the interim PARQUET files
      if they already exist. Defaults to False
      overwrite_output (Optional[bool]): Whether to overwrite the final preprocessed
      files if they already exist. Defaults to False.
    """
    input_filepath = (
        input_filepath or "../../data/raw/exercise_data.50.0000_multionly.mat"
    )
    interim_path = interim_path or "../../data/interim/raw/"
    output_path = output_path or "../../data/interim/preprocessed/"

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    if overwrite_interim:
        logger.info("function will overwrite existing interim data")

    if "multi" in input_filepath:
        logger.info("converting multi-acivity .mat file to PARQUET format")
        logger.info("Loading %s", input_filepath)
        multi_convert_mat_to_parquet(input_filepath, interim_path, overwrite_interim)
    else:
        raise ValueError("Function not defined to preprocess single-activity dataset.")

    logger.info("conversion to PARQUET complete")
    logger.info("creating preprocessed dataset from PARQUET files")
    if overwrite_output:
        logger.info("function will overwrite existing preprocessed data")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

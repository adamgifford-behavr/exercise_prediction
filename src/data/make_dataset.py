# -*- coding: utf-8 -*-
"""
This module defines the functions and methods that create the raw PARQUET files from the
raw data MATLAB files. The `main()` function takes 5 arguments, input_filepath,
output_filepath, val_crit_filepath, test_crit_filepath, and overwrite_output.
...

Args:
      input_filepath (str): location of the raw .mat file. Defaults to
      "../../data/raw/exercise_data.50.0000_multionly.mat"
      output_path (str): location to store the raw PARQUET files. Defaults to
      "../../data/interim/"
      val_crit_filepath (str): path to JSON specifying parameters for splitting
      training files into train vs. validation set. See `_make_train_val_split` for
      all parameter options. Defaults to "./val_split_crit.json".
      test_crit_filepath (str): path to JSON specifying parameters for splitting
      files into train-val vs. test set. See `_make_train_test_sim_split` for
      all parameter options.  Defaults to "./test_split_crit.json".
      overwrite_output (bool): Whether to overwrite the raw PARQUET files
      if they already exist. Defaults to False
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Tuple, Union

import click
import numpy as np
import pandas as pd
import scipy.io as sio
from dotenv import find_dotenv, load_dotenv


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


def _write_json(file_path: Union[str, Path], data: dict) -> None:
    """
    It writes a dictionary to a JSON file

    Args:
      file_path (Union[str, Path]): The path to the file you want to write to.
      data (dict): The data to be written to the file.
    """
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile)


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

    activity_groupings = _read_json(ACTIVITY_GROUPS_FILE)

    # want to reverse relationship between keys and values for easy lookup by activity
    groupings_actvitiy = _reverse_dict(activity_groupings)
    df["label_group"] = df["label"].apply(
        lambda activity: groupings_actvitiy.get(activity, None)
    )

    # ran into unknown issues with pyarrow, but fastparquet seems to work
    df.to_parquet(interim_filepath, engine="fastparquet")


def _fix_activity_groupings(activity_groupings: dict) -> dict:
    all_acts = [v_ix for val in activity_groupings.values() for v_ix in val]
    # there are a couple of duplicated activities
    all_acts = sorted(list(set(all_acts)))

    fixed_groupings = defaultdict(list)
    fixed_groupings["Non-exercise"] = ["Non-Exercise", "Device on Table", "Rest"]
    fixed_groupings["Machines"] = ["Elliptical machine", "Rowing machine", "Cycling"]
    fixed_groupings["Junk"] = ["Arm Band Adjustment", "Arm straight up"]

    used_acts = []
    used_acts.extend(fixed_groupings["Non-exercise"])
    used_acts.extend(fixed_groupings["Machines"])
    used_acts.extend(fixed_groupings["Junk"])
    for act in all_acts:
        if "left" in act:
            fixed_groupings["Left-hand repetitive exercise"].append(act)
        elif "right" in act:
            fixed_groupings["Right-hand repetitive exercise"].append(act)
        elif "Running" in act:
            fixed_groupings["Run"].append(act)
        elif "tretch" in act:
            fixed_groupings["Stretching"].append(act)
        elif ("Walk" in act) and ("lunge" not in act):
            fixed_groupings["Walk"].append(act)
        elif ("Plank" in act) or ("Boat" in act) or act == "Wall Squat":
            fixed_groupings["Static exercise"].append(act)
        elif "lunge" in act.lower():
            fixed_groupings["Repetitive non-arm exercise"].append(act)
        elif (
            ("Unl" in act)
            or (act == "Note")
            or ("Tap" in act)
            or ("Act" in act)
            or ("Inv" in act)
        ):
            fixed_groupings["Junk"].append(act)
        elif act not in used_acts:
            fixed_groupings["Repetitive exercise"].append(act)

    return fixed_groupings


def _write_activity_groupings_json(useful_activity_groupings: np.ndarray) -> None:
    """
    It takes the useful activity groupings data and writes them to a JSON file

    Args:
    usefulActivityGroupings (np.ndarray): a 2D numpy array of the form:
        `array([group_name, array([labels for this group]]))`
    """
    activity_groupings = {row[0]: row[1].tolist() for row in useful_activity_groupings}
    activity_groupings = _fix_activity_groupings(activity_groupings)
    _write_json(ACTIVITY_GROUPS_FILE, activity_groupings)


def _make_files_dict(all_files: list) -> Dict[str, list]:
    """
    It takes a list of file paths and returns a dictionary of subject names and the list of
    data id numbers associated with each subject

    Args:
      all_files (list): list of all files in the directory

    Returns:
      A dictionary with the subject as the key and the list of data ids as the value.
    """
    files_dict = defaultdict(list)
    for file in all_files:
        # ignore non-data files
        if not file.parts[-1].startswith("fileID"):
            continue
        _, subj, _ = file.parts[-1].split("_")
        files_dict[subj].append(str(file))

    return files_dict


def _get_first_subjs_match_crit(
    files_dict: Dict[str, list], n_sing_file: int, n_double_file: int
) -> list:
    """
    It takes a dictionary of subject IDs and a list of files associated with each subject
    ID, and returns a list of the first "n" subject IDs that match the criteria of having
    either one (`n_sing_file`) or two (`n_double_file`) data files associated with them

    Args:
      files_dict (Dict[str, list]): a dictionary of subject IDs and the list of
      file ids they have.
      n_sing_file (int): number of subjects with only one file to include in the test
      dataset
      n_double_file (int): number of subjects with two files to include in the test
      dataset

    Returns:
      A list of the first subject IDs that match the criteria.
    """
    test_subj_ids = []
    ones_left = n_sing_file
    twos_left = n_double_file
    for key, val in files_dict.items():
        if not (ones_left or twos_left):
            break

        if (len(val) == 1) & (ones_left > 0):
            test_subj_ids.append(key)
            ones_left -= 1
        elif (len(val) == 2) & (twos_left > 0):
            test_subj_ids.append(key)
            twos_left -= 1

    return test_subj_ids


def _make_train_test_dict(
    all_files: List[Path], test_subj_ids: list, sim_subj_ids: list
) -> Dict[str, list]:
    """
    It takes a dictionary of file paths and splits them into three lists, one for training,
    one for testing, and one for deployment simulation.

    Args:
      all_files (Dict[str, list]): a list of all the files in the directory
      test_subj_ids (list): list of subject IDs to use for testing

    Returns:
      A dictionary with two keys: "test" and "train_val". The values are lists of strings
      specifying the file paths of the train and test data.
    """
    train_test_files = defaultdict(list)
    for file in all_files:
        _, subj, _ = file.parts[-1].split("_")
        if any(subj == test_subj for test_subj in test_subj_ids):
            train_test_files["test"].append(str(file))
        elif any(subj == sim_subj for sim_subj in sim_subj_ids):
            train_test_files["simulate"].append(str(file))
        else:
            train_test_files["train_val"].append(str(file))

    return train_test_files


def _make_train_test_sim_split(
    interim_path: Path, n_sing_file: int, n_double_file: int, n_sim_file: int = 1
) -> Dict[str, List[str]]:
    """
    It takes a directory of files, and splits them into train and test sets, based on the
    number of subjects in the test set that should have one (`n_file_single) or two
    (`n_file_double`) data recording files. This function simply finds the first subjects
    to meet these criteria by cycling through the files in `interim_path` in alphanumeric
    order. It then subsequently takes n_sim_file from the test list to save for
    simulating batch and/or stream service predictions and model updating. This function
    simply grabs the last files from the test dataset list to save for streaming.

    Args:
      interim_path (Path): the path to the interim data folder
      n_sing_file (int): number of single-data-file subjects to include in the test set
      n_double_file (int): number of double-data-file subjects to include in the test set
      n_sim_file (int): number of files from the test dataset to save for simulating
      real-world deployment and module updating,
    """
    all_files = sorted(
        list(x for x in interim_path.iterdir() if x.is_file() and "fileID" in str(x))
    )
    files_dict = _make_files_dict(all_files)

    test_subj_ids = _get_first_subjs_match_crit(files_dict, n_sing_file, n_double_file)
    sim_subj_ids = []
    for _ in range(n_sim_file):
        sim_subj_ids.append(test_subj_ids.pop())

    train_test_files = _make_train_test_dict(all_files, test_subj_ids, sim_subj_ids)

    # _write_json(TRAIN_TEST_FILE, train_test_files)
    return train_test_files


def write_single_parquet_file_wrapper(
    interim_path: Union[str, Path],
    subj_data: sio.matlab.mat_struct,
    data_id: int = 0,
    overwrite: bool = False,
):
    """
    Wrapper function that takes an array of subject data, calls `_write_single_parquet_file`
    to write it to a parquet file, and returns nothing

    Args:
      interim_path (str): full interim file path to save the parquet file
      subj_data (sio.matlab.mat_struct): array of subject data
      data_id (int): int = 0,. Defaults to 0
      overwrite (bool): bool = False,. Defaults to False
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


def _get_desired_multifile_params(**kwargs) -> dict:
    """
    Gets desired number of subjects to include in validation set that have 5, 4, 3, and
    2 data-recording files, respectively using values passed in kwargs combined with
    defaults. Will fill in any additional subjects to meet desired subject and file count
    with those subjects that have only one recording file, so we don't include this
    criterion here

    Returns:
      A dictionary with the keys: n_5_files, n_4_files, n_3_files, n_2_files. The values
      are the number of files that should be used for validation for each of the keys.
    """
    default_ns = {"n_5_files": 0, "n_4_files": 1, "n_3_files": 1, "n_2_files": 1}
    desired_multifile_counts = {
        key: val if key not in kwargs else kwargs[key]
        for key, val in default_ns.items()
    }
    desired_multifile_counts.update(
        {
            key: val
            for key, val in kwargs.items()
            if key not in desired_multifile_counts.keys()
        }
    )
    return desired_multifile_counts


def _get_sorted_train_val_files(files_dict: dict) -> Tuple[list, list]:
    """
    Sort the subjects by the number of files they have, and return the sorted list of
    subjects and the sorted list of file counts

    Args:
      files_dict (dict): a dictionary of the form {subject_id: [list of files]}

    Returns:
      A tuple of two lists.
    """
    subjs_file_counts = {key: len(val) for key, val in files_dict.items()}

    sort_subjs_counts = sorted(
        subjs_file_counts.items(), key=lambda x: x[1], reverse=True
    )
    sorted_subjs, sorted_file_counts = (
        [s[0] for s in sort_subjs_counts],
        [s[1] for s in sort_subjs_counts],
    )
    return sorted_file_counts, sorted_subjs


def do_initial_split(
    desired_multifile_counts: dict,
    desired_files_remaining: int,
    sorted_files: list,
    sorted_subjs: list,
) -> Tuple[Tuple[list, list], Tuple[list, list], int]:
    """
    This function does an initial split of the files determined by the data in
    `sorted_subjs` and `sorted_files`  into training vs. validation by
    attempting to achieve the constraints in `desired_multifile_counts`,
    `desired_files_remaining` (total number of files to include in validation set).

    Args:
      desired_multifile_counts (dict): dictionary of constraints defining the number of
      subjects to include that have 5, 4, 3, and 2 data recordings, respectively
      desired_files_remaining (int): The number of files we want to include in the
      validation split.
      sorted_files (list): the sorted list of data-recording file counts
      sorted_subjs (list): list of subject IDs sorted by data-recording file counts

    Returns:
      A tuple of two tuples and an int:
        (val_files, val_subjs),
        (sorted_files, sorted_subjs),
        desired_files_remaining
    """
    logger = logging.getLogger(__name__)

    val_subjs, val_files = [], []
    for key, desired_count in desired_multifile_counts.items():
        desired_n = int(key.split("_")[1])
        for n in range(desired_count):
            try:
                n_ix = sorted_files.index(desired_n)
            except ValueError:
                logger.error(
                    "Couldn't find subject %d with desired count %d in "
                    "the files left over for validation split. Please reduce the requested"
                    " number of subjects for this desired count and re-run.",
                    n,
                    desired_count,
                )
                raise

            val_subjs.append(sorted_subjs.pop(n_ix))
            val_files.append(sorted_files.pop(n_ix))
            desired_files_remaining -= val_files[-1]

    return (val_files, val_subjs), (sorted_files, sorted_subjs), desired_files_remaining


def _data_tuple_from_dict(data_dict: dict) -> Generator:
    """
    It takes a dictionary and returns a tuple of the values in the dictionary, sorted by the
    keys

    Args:
      data_dict (dict): a dictionary of data

    Returns:
      A generator object.
    """
    return (data_dict[key] for key in sorted(data_dict.keys()))


def _attempt_satisfy_down(
    val_data: dict, desired_data: dict, from_data: dict, tol: int
) -> Tuple[list, list]:
    """
    Attempts to fix the initial train-val split done by `do_initial_split` to reduce the
    number of subjects in validation set to match the desired number defined in
    `desired_data`. Errors if it can't easily swap out single-data-file subjects from
    training set to validation set without violating tolerance on number of total files
    in validation set.

    NOTE: there are probably more sophisticated ways to satisfy these constraints (e.g.,
    swapping subjects that have more than one recording file), but for simplicity we have
    ignored this potential. Future work can build a more robust system for satisfying
    constraints, if desired.

    Args:
      val_data (dict): dict with fields "file_counts", "subjects"
      desired_data (dict): dict with fields "file_counts", "subjects"
      from_data (dict): dict with fields "file_counts", "subjects"
      tol (int): tolerance for how far we can be from desired number of total data files
      in validation set

    Returns:
      A tuple of two lists, splitting validation and training subjects.
    """
    logger = logging.getLogger(__name__)

    val_files, val_subjs = _data_tuple_from_dict(val_data)
    desired_val_files, desired_val_subjs = _data_tuple_from_dict(desired_data)
    sorted_files, sorted_subjs = _data_tuple_from_dict(from_data)

    while (tol > 0) or (len(val_subjs) > desired_val_subjs):
        # popping from end, which removes subjects with 1 data file first
        sorted_subjs.append(val_subjs.pop())
        sorted_files.append(val_files.pop())
        tol -= sorted_files[-1]

    if len(val_subjs) > desired_val_subjs:
        logger.error(
            "Cannot reconcile requirements for validation subject counts having "
            "particular numbers of data files with the constraints on number of "
            "validation subjects %d, number of total validation files "
            "%d and validation file count tolerance %d. Please"
            " either adjust subject counts by number of data files, the total number"
            " of desired validation files, or increase file count tolerance.",
            desired_val_subjs,
            desired_val_files,
            tol,
        )
        raise RuntimeError()

    logger.info(
        "Fixed total-validation subject constraint given file-count tolerance..."
    )
    return val_subjs, sorted_subjs


def _attempt_satisfy_up(
    val_data: dict, desired_data: dict, from_data: dict, tol: int
) -> Tuple[list, list]:
    """
    Attempts to fix the initial train-val split done by `do_initial_split` to increase the
    number of subjects in validation set to match the desired number defined in
    `desired_data`. Errors if it can't easily swap out single-data-file subjects from
    validation set to training set without violating tolerance on number of total files
    in validation set.

    NOTE: there are probably more sophisticated ways to satisfy these constraints (e.g.,
    swapping subjects that have more than one recording file), but for simplicity we have
    ignored this potential. Future work can build a more robust system for satisfying
    constraints, if desired.

    Args:
      val_data (dict): dict with fields "file_counts", "subjects"
      desired_data (dict): dict with fields "file_counts", "subjects"
      from_data (dict): dict with fields "file_counts", "subjects"
      tol (int): tolerance for how far we can be from desired number of total data files
      in validation set

    Returns:
      A tuple of two lists, splitting validation and training subjects.
    """

    logger = logging.getLogger(__name__)

    val_files, val_subjs = _data_tuple_from_dict(val_data)
    desired_val_files, desired_val_subjs = _data_tuple_from_dict(desired_data)
    sorted_files, sorted_subjs = _data_tuple_from_dict(from_data)

    while (tol > 0) or (len(val_subjs) < desired_val_subjs):
        # popping from end, which removes subjects with 1 data file first
        val_subjs.append(sorted_subjs.pop())
        val_files.append(sorted_files.pop())
        tol -= val_files[-1]

    if len(val_subjs) > desired_val_subjs:
        logger.error(
            "Cannot reconcile requirements for validation subject counts having "
            "particular numbers of data files with the constraints on number of "
            "validation subjects %d, number of total validation files "
            "%d and validation file count tolerance %d. Please"
            " either adjust subject counts by number of data files, the total number"
            " of desired validation files, or increase file count tolerance.",
            desired_val_subjs,
            desired_val_files,
            tol,
        )
        raise RuntimeError()

    logger.info(
        "Fixed total-validation subject constraint given file-count tolerance..."
    )
    return val_subjs, sorted_subjs


def verify_satisfy_split(
    val_data: dict, desired_data: dict, from_data: dict, tol: int
) -> Tuple[list, list]:
    """
    This function attempts to verify that the initial train-val split done in
    `do_initial_split` satisfies all constrains on desired number of subjects in
    validation set as well as total number of files in validation set. It calls one of
    two functions if, after the initial sort, there are too many or two few subjects in
    the validation set. Those functions either fix the issue and satisfy constraints, or
    error out.

    Args:
      val_data (dict): dict with fields "file_counts", "subjects"
      desired_data (dict): dict with fields "file_counts", "subjects"
      from_data (dict): dict with fields "file_counts", "subjects"
      tol (int): tolerance for how far we can be from desired number of total data files
      in validation set

    Returns:
      a tuple of two lists.
    """
    logger = logging.getLogger(__name__)

    val_files, val_subjs = _data_tuple_from_dict(val_data)
    (
        desired_val_files,
        desired_files_remaining,
        desired_val_subjs,
    ) = _data_tuple_from_dict(desired_data)
    sorted_files, sorted_subjs = _data_tuple_from_dict(from_data)

    while desired_files_remaining > 0:
        val_subjs.append(sorted_subjs.pop())
        desired_files_remaining -= sorted_files.pop()

    val_data = {"file_counts": val_files, "subjects": val_subjs}
    desired_data = {"file_counts": desired_val_files, "subjects": desired_val_subjs}
    from_data = {"file_counts": sorted_files, "subjects": sorted_subjs}
    if len(val_subjs) > desired_val_subjs:
        logger.info(
            "Too many subjects selected based in initial passthrough. Attempting to fix..."
        )

        val_subjs, train_subjs = _attempt_satisfy_down(
            val_data, desired_data, from_data, tol
        )

    elif len(val_subjs) < desired_val_subjs:
        logger.info(
            "Not enough subjects selected based in initial passthrough. Attempting to fix..."
        )

        val_subjs, train_subjs = _attempt_satisfy_up(
            val_data, desired_data, from_data, tol
        )

    return val_subjs, train_subjs


def _make_train_val_split(
    train_test_files: Dict[str, List[str]],
    desired_val_subjs: int,
    desired_val_files: int,
    n_files_tol: int = 1,
    **kwargs,
) -> Dict[str, List[str]]:
    # pylint: disable=too-many-locals
    """
    It takes in a dictionary splitting data into train/val, test, and simulation, and the
    desired number of subjects and files to include in the validation set,
    and then it splits the train/val set into a validation set and a training set, such that
    the validation set has the desired number of subjects and files, with the added
    constraints regarding the number of subjects included that have certain numbers of
    data files defined by kwargs (see `_get_desired_multifile_params`).

    Args:
      train_test_files (Dict[str, List[str]]): a dictionary that contains the list of files
      per uber data category (i.e., keys: "train_val", "test", and "sim")
      desired_val_subjs (int): the number of subjects you want in the validation set
      desired_val_files (int): the number of files you want in the validation set
      n_files_tol (int): the number of files that can be off from the desired
      number of files in the validation set. Defaults to 1
    """
    # this loads in the desired number of subjects with multiple data recordings to
    # include in the validation set
    desired_multifile_counts = _get_desired_multifile_params(**kwargs)

    # here, we organize the subjects by decreasing data-recording counts to use in
    # systematizing the segregation of subjects & files such that this process is
    # idempotent
    # since we first make the split between train/val & test, we load in this data
    # and then just focus on the train files to split further
    train_val_files = [Path(file) for file in train_test_files["train_val"]]
    files_dict = _make_files_dict(train_val_files)
    sorted_data = _get_sorted_train_val_files(files_dict)

    # initial split into val and train
    val_tup, train_tup, desired_files_remaining = do_initial_split(
        desired_multifile_counts, desired_val_files, *sorted_data
    )

    # verify and/or attempt to fix split to satisfy constraints
    val_data = {"file_counts": val_tup[0], "subjects": val_tup[1]}
    desired_data = {
        "file_counts": desired_val_files,
        "files_remaining": desired_files_remaining,
        "subjects": desired_val_subjs,
    }
    from_data = {"file_counts": train_tup[0], "subjects": train_tup[1]}
    val_subjs, train_subjs = verify_satisfy_split(
        val_data, desired_data, from_data, n_files_tol
    )

    # now, find the file names for each subject in val_subjs and train_subjs and store as dict
    trn_val_tst_sim_dict = {
        "validation": [file for subj in val_subjs for file in files_dict[subj]],
        "train": [file for subj in train_subjs for file in files_dict[subj]],
    }

    # # now, write dict to json
    # _write_json(TRAIN_VAL_FILE, train_val_dict)
    trn_val_tst_sim_dict["test"] = train_test_files["test"]
    trn_val_tst_sim_dict["simulate"] = train_test_files["simulate"]
    return trn_val_tst_sim_dict


def make_data_splits_json(
    interim_path: str,
    test_split_criteria: Dict[str, int],
    val_split_criteria: Dict[str, int],
    overwrite_output: bool,
) -> None:
    """
    This function akes in the path to the interim data folder, the test split criteria,
    and the validation split criteria, and creates two json files in the interim data
    folder: one that splits the data into train/val vs. test, and one that splits the
    data into train vs. val.

    The function first checks if the train/val vs. test split file exists. If it doesn't,
    it calls the `_make_train_test_sim_split` function, which takes in the path to the
    interim data folder and the test split criteria, and creates the train/val vs. test
    split file.

    The function then checks if the train vs. val split file exists. If it doesn't, it
    calls the `_make_train_val_split` function, which takes in the validation split
    criteria and creates the train vs. val split file.

    Args:
      interim_path (str): the path to the interim data folder
      test_split_criteria (Dict[str, int]): Criterion for splitting data between test
      and train/val datasets, in the form of:
        {
            "n_double_file": int,
            "n_sing_file": int
        }
      where "n_double_file" is the number of subjects that have 2 data recordings, and
      "n_sing_file" is the number of subjects that have 1 data recording
      val_split_criteria (Dict[str, int]): Criterion for splitting data between test
      and train and val datasets, in the form of:
        {
            "desired_val_files": int,
            "desired_val_subjs": int,
            "n_2_files": int,
            "n_3_files": int,
            "n_4_files": int,
            "n_5_files": int,
            "n_files_tol": int
        }
      where "desired_val_files" is the total number of files in the validation set,
      "desired_val_subjs" is the total number of subjects in the validation set,
      "n_2_files" is the number of subjects with 2 data recordings in the validation set,
      "n_3_files" is the number of subjects with 3 data recordings in the validation set,
      "n_4_files" is the number of subjects with 4 data recordings in the validation set,
      "n_5_files" is the number of subjects with 5 data recordings in the validation set,
      "n_files_tol" is the tolerance on the total number of files in the validation set,
      used to try to satisfy all constraints.
      overwrite_output (bool): Whether to overwrite any preexisting DATA_GROUP_SPLITS
      json file
    """
    logger = logging.getLogger(__name__)
    if (not DATA_GROUP_SPLITS.exists()) or overwrite_output:
        logger.info(
            (
                "json file for splitting files into groups (e.g., train, test, etc.) does "
                "not exist or overwrite setting is on. writing file first..."
            )
        )
        train_test_files = _make_train_test_sim_split(
            Path(interim_path), **test_split_criteria
        )
        trn_val_tst_sim_splits = _make_train_val_split(
            train_test_files, **val_split_criteria
        )
        _write_json(DATA_GROUP_SPLITS, trn_val_tst_sim_splits)
        logger.info("writing datafile group splits complete")
    else:
        logger.info("json file for splitting files into groups already exists...")


@click.command()
@click.argument(
    "input_filepath",
    type=click.Path(exists=True),
    required=False,
    default="../../data/raw/exercise_data.50.0000_multionly.mat",
)
@click.argument(
    "output_path",
    type=click.Path(exists=True),
    required=False,
    default="../../data/interim/",
)
@click.argument(
    "val_crit_filepath",
    type=click.Path(exists=True),
    required=False,
    default="./val_split_crit.json",
)
@click.argument(
    "test_crit_filepath",
    type=click.Path(exists=True),
    required=False,
    default="./test_split_crit.json",
)
@click.option(
    "--overwrite-output/--keep-output",
    type=bool,
    default=False,
    show_default=True,
    help=(
        "Whether to overwrite files of matching output data by filename if they already "
        "exist."
    ),
)
def main(
    input_filepath: str = "../../data/raw/exercise_data.50.0000_multionly.mat",
    output_path: str = "../../data/interim/",
    val_crit_filepath: str = "./val_split_crit.json",
    test_crit_filepath: str = "./test_split_crit.json",
    overwrite_output: bool = False,
) -> None:
    # pylint: disable=too-many-arguments
    """
    Runs data processing scripts to turn raw data from (../raw) in MATLAB format into
    PARQUET files ready to be featurized (saved in ../interim/). Function also creates
    JSON files which serve to separate data files into train, validation, and test sets.

    Args:
      input_filepath (str): location of the raw .mat file. Defaults to
      "../../data/raw/exercise_data.50.0000_multionly.mat"
      output_path (str): location to store the raw PARQUET files. Defaults to
      "../../data/interim/"
      val_crit_filepath (str): path to JSON specifying parameters for splitting
      training files into train vs. validation set. See `_make_train_val_split` for
      all parameter options. Defaults to "./val_split_crit.json".
      test_crit_filepath (str): path to JSON specifying parameters for splitting
      files into train-val vs. test set. See `_make_train_test_sim_split` for
      all parameter options.  Defaults to "./test_split_crit.json".
      overwrite_output (bool): Whether to overwrite the raw PARQUET files
      if they already exist. Defaults to False
    """
    logger = logging.getLogger(__name__)
    logger.info("making interim dataset from raw data")

    if overwrite_output:
        logger.info("function will overwrite existing interim data")

    if "multi" in input_filepath:
        logger.info("converting multi-acivity .mat file to PARQUET format")
        logger.info("Loading %s", input_filepath)
        multi_convert_mat_to_parquet(input_filepath, output_path, overwrite_output)
    else:
        raise ValueError("Function not defined to preprocess single-activity dataset.")

    logger.info("conversion to PARQUET complete")

    logger.info("creating train-val-test split JSONs")
    test_split_criteria = _read_json(test_crit_filepath)
    val_split_criteria = _read_json(val_crit_filepath)
    # remove comment from json data if it exists
    val_split_criteria.pop("_comment", None)
    make_data_splits_json(
        output_path, test_split_criteria, val_split_criteria, overwrite_output
    )
    logger.info("dataset creation complete...")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[1]
    ACTIVITY_GROUPS_FILE = project_dir / "data" / "activity_groupings.json"
    DATA_GROUP_SPLITS = project_dir / "features" / "datafile_group_splits.json"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

import json
from pathlib import Path

from deepdiff import DeepDiff

import src.data.make_dataset as md

PROJDIR = Path(__file__).resolve().parents[1]
md.ACTIVITY_GROUPS_FILE = PROJDIR / "tests/activity_groupings.json"
md.DATA_GROUP_SPLITS = PROJDIR / "tests/datafile_group_splits.json"


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data


## src/data/make_dataset.py
# writing activity groupings json file
def test_write_activity_groupings_json():
    input_filepath = PROJDIR / "data/raw/exercise_data.50.0000_multionly.mat"
    mat_contents = md._load_matfile(input_filepath)

    md._write_activity_groupings_json(
        mat_contents["exerciseConstants"].usefulActivityGroupings
    )
    actual_result = read_json(md.ACTIVITY_GROUPS_FILE)
    expected_result = read_json(PROJDIR / "src/data/activity_groupings.json")

    diff = DeepDiff(expected_result, actual_result)
    print(f"diff={diff}")

    assert "values_changed" not in diff, "values_changed"
    assert "iterable_item_added" not in diff, "iterable_item_added"
    assert "iterable_item_removed" not in diff, "iterable_item_removed"
    assert "dictionary_item_added" not in diff, "dictionary_item_added"
    assert "dictionary_item_removed" not in diff, "dictionary_item_removed"

    # delete test file after
    md.ACTIVITY_GROUPS_FILE.unlink()


# train test sim split
def test_make_data_splits():
    interim_path = PROJDIR / "data/interim"

    test_split_criteria = dict(n_double_file=6, n_sim_file=1, n_sing_file=13)

    val_split_criteria = dict(
        desired_val_files=20,
        desired_val_subjs=15,
        n_2_files=1,
        n_3_files=1,
        n_4_files=1,
        n_5_files=0,
        n_files_tol=1,
    )
    md.make_data_splits_json(interim_path, test_split_criteria, val_split_criteria)
    actual_result = read_json(md.DATA_GROUP_SPLITS)
    expected_result = read_json(PROJDIR / "src/features/datafile_group_splits.json")

    # can't test the dicts exactly because expected_result uses relative paths from
    # src/data and actual_result uses absolute paths via Path()
    for key, files in actual_result.items():
        for file in files:
            filename = file.split("\\")[-1]
            # this checks that all files in each actual key are in the respected expected
            # key
            assert any(
                filename in exp_files for exp_files in expected_result[key]
            ), f"{filename} expected in {key} but not found"

    all_actual_files = [fix for val in actual_result.values() for fix in val]
    all_expected_files = [fix for val in expected_result.values() for fix in val]
    assert len(all_actual_files) == len(
        all_expected_files
    ), "number of files does not match"

    # delete test file after
    md.DATA_GROUP_SPLITS.unlink()

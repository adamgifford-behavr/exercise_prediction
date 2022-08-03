# -*- coding: utf-8 -*-
"""
This module defines the functions and methods that create the datasets from the raw data
files. The `main()` function takes two arguments, input_filepath and output_filepath, and
...

Args:
    input_filepath: The path to the input file.
    output_filepath: The path to the output file.
"""
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    It takes in a filepath, does nothing with it, and then logs a message.

    Args:
      input_filepath: The path to the input file.
      output_filepath: /kaggle/working/
    """
    if input_filepath:
        pass
    if output_filepath:
        pass

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(None, None)

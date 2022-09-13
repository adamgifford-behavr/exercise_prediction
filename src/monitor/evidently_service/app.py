#!/usr/bin/env python3
# pylint: disable=no-name-in-module
# pylint: disable=import-error
# pylint: disable=implicit-str-concat
"""
This is a demo service for Evidently metrics integration with Prometheus and Grafana.

Read `README.md` for proper setup and installation.

The service gets a reference dataset from reference.csv file and process current data
with HTTP API.

Metrics calculation results are available with `GET /metrics` HTTP method in Prometheus
compatible format.
"""
import dataclasses
import datetime
import logging
import os
import sys
from typing import Dict, List, Optional

import flask
import pandas as pd
import prometheus_client
import yaml
from evidently.model_monitoring import ClassificationPerformanceMonitor  # NOQA
from evidently.model_monitoring import ModelMonitoring  # NOQA
from evidently.model_monitoring import CatTargetDriftMonitor, DataDriftMonitor  # NOQA
from evidently.pipeline.column_mapping import ColumnMapping
from flask import Flask
from pyarrow import parquet as pq
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# Add prometheus wsgi middleware to route /metrics requests
app.wsgi_app = DispatcherMiddleware(
    app.wsgi_app, {"/metrics": prometheus_client.make_wsgi_app()}
)


@dataclasses.dataclass
class MonitoringServiceOptions:
    """Class that defines monitoring service options"""

    datasets_path: str
    min_reference_size: int
    use_reference: bool
    moving_reference: bool
    window_size: int
    calculation_period_sec: int


@dataclasses.dataclass
class LoadedDataset:
    """Class that defines the loaded dataset"""

    name: str
    references: pd.DataFrame
    monitors: List[str]
    column_mapping: ColumnMapping


EVIDENTLY_MONITORS_MAPPING = {
    "cat_target_drift": CatTargetDriftMonitor,
    "data_drift": DataDriftMonitor,
    "classification_performance": ClassificationPerformanceMonitor,
}


class MonitoringService:  # pylint: disable=too-few-public-methods
    """Class that defines the monitoring service"""

    # names of monitoring datasets
    datasets: List[str]
    metric: Dict[str, prometheus_client.Gauge]
    last_run: Optional[datetime.datetime]
    # collection of reference data
    reference: Dict[str, pd.DataFrame]
    # collection of current data
    current: Dict[str, Optional[pd.DataFrame]]
    # collection of monitoring objects
    monitoring: Dict[str, ModelMonitoring]
    calculation_period_sec: float = 15
    window_size: int

    def __init__(self, datasets: Dict[str, LoadedDataset], window_size: int) -> None:
        """
        The function takes a dictionary of datasets, and a window size, and creates a
        reference, monitoring, and column mapping dictionary for each dataset, and
        creates a metrics and next run time dictionary

        Args:
          datasets (Dict[str, LoadedDataset]): Dict[str, LoadedDataset]
          window_size (int): The number of rows to process at a time.
        """
        self.reference = {}
        self.monitoring = {}
        self.current = {}
        self.column_mapping = {}
        self.window_size = window_size

        for dataset_info in datasets.values():
            self.reference[dataset_info.name] = dataset_info.references
            self.monitoring[dataset_info.name] = ModelMonitoring(
                monitors=[
                    EVIDENTLY_MONITORS_MAPPING[k]() for k in dataset_info.monitors
                ],
                options=[],
            )
            self.column_mapping[dataset_info.name] = dataset_info.column_mapping

        self.metrics = {}
        self.next_run_time = {}

    def iterate(self, dataset_name: str, new_rows: pd.DataFrame):
        """
        If the dataset name is in the current dictionary, append the new rows to the current
        dataframe, otherwise, set the current dataframe to the new rows.

        Args:
          dataset_name (str): The name of the dataset to be monitored.
          new_rows (pd.DataFrame): the dataframe that is being passed in

        Returns:
          The return value is a list of tuples. Each tuple contains the following:
        """
        window_size = self.window_size

        if dataset_name in self.current:
            current_data = self.current[dataset_name].append(
                new_rows, ignore_index=True
            )

        else:
            current_data = new_rows

        current_size = current_data.shape[0]

        if current_size > self.window_size:
            # cut current_size by window size value
            current_data.drop(
                index=list(range(0, current_size - self.window_size)), inplace=True
            )
            current_data.reset_index(drop=True, inplace=True)

        self.current[dataset_name] = current_data

        if current_size < window_size:
            logging.info(
                "Not enough data for measurement: %s of %s."
                " Waiting more data",  # pylint: disable=implicit-str-concat
                current_size,
                window_size,
            )
            return

        next_run_time = self.next_run_time.get(dataset_name)

        if next_run_time is not None and next_run_time > datetime.datetime.now():
            logging.info("Next run for dataset %s at %s", dataset_name, next_run_time)
            return

        self.next_run_time[dataset_name] = datetime.datetime.now() + datetime.timedelta(
            seconds=self.calculation_period_sec
        )
        self.monitoring[dataset_name].execute(
            self.reference[dataset_name],
            current_data,
            self.column_mapping[dataset_name],
        )

        for metric, value, labels in self.monitoring[dataset_name].metrics():
            metric_key = f"evidently:{metric.name}"
            found = self.metrics.get(metric_key)

            if not labels:
                labels = {}

            labels["dataset_name"] = dataset_name

            if isinstance(value, str):
                continue

            if found is None:
                found = prometheus_client.Gauge(
                    metric_key, "", list(sorted(labels.keys()))
                )
                self.metrics[metric_key] = found

            try:
                found.labels(**labels).set(value)

            except ValueError as error:
                # ignore errors sending other metrics
                logging.error("Value error for metric %s, error: %s", metric_key, error)


SERVICE: Optional[MonitoringService] = None


@app.before_first_request
def configure_service():
    """
    It reads a config file, loads the reference data, and creates a MonitoringService
    object
    """
    # pylint: disable=global-statement
    global SERVICE
    config_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )

    # try to find a config file, it should be generated via the data preparation script
    if not os.path.exists(config_file_path):
        logging.error("File %s does not exist", config_file_path)
        sys.exit(
            (
                "Cannot find a config file for the metrics service. Try to check README.md"
                " for setup instructions."
            )
        )

    with open(config_file_path, "rb") as config_file:
        config = yaml.safe_load(config_file)

    options = MonitoringServiceOptions(**config["service"])
    datasets = {}

    for dataset_name, dataset_options in config["datasets"].items():
        reference_file = dataset_options["reference_file"]
        logging.info(
            "Load reference data for dataset %s from %s", dataset_name, reference_file
        )
        reference_data = pq.read_table(reference_file).to_pandas()
        reference_data = reference_data.dropna(subset=["label_group"])
        datasets[dataset_name] = LoadedDataset(
            name=dataset_name,
            references=reference_data,
            monitors=dataset_options["monitors"],
            column_mapping=ColumnMapping(**dataset_options["column_mapping"]),
        )
        logging.info(
            "Reference is loaded for dataset %s: %s rows",
            dataset_name,
            len(reference_data),
        )

    SERVICE = MonitoringService(datasets=datasets, window_size=options.window_size)


@app.route("/iterate/<dataset>", methods=["POST"])
def iterate(dataset: str):
    """
    It takes a JSON object, converts it to a Pandas DataFrame, and then passes it to the
    iterate function of the service.

    Args:
      dataset (str): str

    Returns:
      a string.
    """
    item = flask.request.json

    global SERVICE  # pylint: disable=global-variable-not-assigned
    if SERVICE is None:
        return "Internal Server Error: service not found", 500

    SERVICE.iterate(dataset_name=dataset, new_rows=pd.DataFrame.from_dict(item))
    return "ok"


if __name__ == "__main__":
    app.run()

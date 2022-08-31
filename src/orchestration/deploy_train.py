"""Creates a deployment in Prefect for the flow `train_flow` in `orchestrate_train.py`."""
import os

from dotenv import find_dotenv, load_dotenv
from prefect.deployments import Deployment

import src.orchestration.orchestrate_train as ot

# from datetime import timedelta
# from prefect.orion.schemas.schedules import IntervalSchedule
# from prefect.filesystems import LocalFileSystem
# from prefect.filesystems import S3


load_dotenv(find_dotenv())


FEATURIZE_ID = os.getenv("FEATURIZE_ID")
EXP_NAME = os.getenv("EXP_NAME", "exercise_prediction_naive_feats")
DEBUG = os.getenv("DEBUG", "false") == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "prod")
FLOW_VERSION = os.getenv("FLOW_VERSION")

if DEBUG:
    EXP_NAME = EXP_NAME + "_debug"

# storage = S3.load("dev-bucket") # load a pre-defined block
# storage = LocalFileSystem.load()

deployment = Deployment.build_from_flow(
    flow=ot.train_flow,
    name="Exercise Group-Naive Features Predictions Training",
    tags=[EXP_NAME, FEATURIZE_ID, DEBUG, ENVIRONMENT],
    version=FLOW_VERSION,
    entrypoint="./",
    # schedule=IntervalSchedule(interval=timedelta(weeks=4)),
    # storage=storage,
    work_queue_name="manual_training_flow",
)
deployment.apply()

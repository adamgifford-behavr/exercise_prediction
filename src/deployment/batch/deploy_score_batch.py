"""Creates a deployment in Prefect for the flow `score_flow` in
`orchestrate_score_batch.py`.
"""
import os

from dotenv import find_dotenv, load_dotenv
from prefect.deployments import Deployment

import src.deployment.batch.orchestrate_score_batch as osb

# from prefect.filesystems import S3
# from datetime import timedelta
# from prefect.orion.schemas.schedules import IntervalSchedule
# from prefect.filesystems import LocalFileSystem

load_dotenv(find_dotenv())


FEATURIZE_ID = os.getenv("FEATURIZE_ID")
EXP_NAME = os.getenv("EXP_NAME", "exercise_prediction_naive_feats_pipe")
DEBUG = os.getenv("DEBUG", "false") == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "prod")
FLOW_VERSION = os.getenv("FLOW_VERSION")

if DEBUG:
    EXP_NAME = EXP_NAME + "_debug"

# storage = S3.load("score-batch-logs")  # load a pre-defined block
# storage = LocalFileSystem.load()

deployment = Deployment.build_from_flow(
    flow=osb.score_flow,
    name="Exercise Group-Naive Features Batch Scoring",
    tags=[EXP_NAME, FEATURIZE_ID, DEBUG, ENVIRONMENT],
    version=FLOW_VERSION,
    entrypoint="./",
    # schedule=IntervalSchedule(interval=timedelta(weeks=4)),
    # storage=storage,
    work_queue_name="manual_scoring_flow",
)
deployment.apply()

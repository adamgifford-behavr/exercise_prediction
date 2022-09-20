# pylint: disable=duplicate-code
# pylint: disable=import-error
"""This module is the lambda function that would be used in AWS lambda for a true
streaming service implementation of the model. It is used in a Docker container
image replicating the AWS lambda service to simulate a streaming prediction service.
"""
import os

import model
from aws_lambda_typing import context as context_
from aws_lambda_typing import events

PREDICTIONS_STREAM_NAME = os.getenv("PREDICTIONS_STREAM_NAME", "predictions-stream")
RUN_ID = os.getenv("RUN_ID")
TEST_RUN = os.getenv("TEST_RUN", "True") == "True"

model_service = model.init(
    prediction_stream_name=PREDICTIONS_STREAM_NAME,
    desired_freqs="frequency_features.json",
    run_id=RUN_ID,
    test_run=TEST_RUN,
)


def lambda_handler(event: events.SQSEvent, context: context_.Context) -> dict:
    """
    It takes an SQS event and returns a response

    Args:
      event (events.SQSEvent): event received from Amazon Kinesis
      context (context_.Context): This is the context object that Lambda provides. It
      contains information about the Lambda function, such as the function name, the memory
      allocated, and the time remaining.

    Returns:
      The output of the lambda_handler function is being returned.
    """
    # pylint: disable=unused-argument
    return model_service.lambda_handler(event)

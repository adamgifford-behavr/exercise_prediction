#!/usr/bin/env bash

if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"
fi

if [ "${LOCAL_IMAGE_NAME}" == "" ]; then
    LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
    export LOCAL_IMAGE_NAME="stream-exercise-prediction:${LOCAL_TAG}"
    echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_IMAGE_NAME}"
    cp -r ../models ../src/deployment/streaming
    docker build -t ${LOCAL_IMAGE_NAME} ../src/deployment/streaming/
else
    echo "no need to build image ${LOCAL_IMAGE_NAME}"
fi

export PREDICTIONS_STREAM_NAME="predictions_stream"

docker-compose up -d

sleep 5

aws --endpoint-url=http://localhost:4566 \
    kinesis create-stream \
    --stream-name ${PREDICTIONS_STREAM_NAME} \
    --shard-count 1

if [[ -z "${GITHUB_ACTIONS}" ]]; then
    cd ../src/deployment/streaming/
    pipenv run python ../../../integration_tests/test_docker.py
    cd ../../../integration_tests/
else
    pipenv run python integration_tests/test_docker.py
fi

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi


cd ../src/deployment/streaming/
pipenv run python ../../../integration_tests/test_kinesis.py
cd ../../../integration_tests/

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi


docker-compose down
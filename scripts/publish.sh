#!/usr/bin/env bash

echo "publishing image ${LOCAL_IMAGE_NAME} to ECR..."

REMOTE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
REMOTE_TAG="latest"
REMOTE_IMAGE=${REMOTE_URI}"/"${REPOSITORY}:${REMOTE_TAG}

aws ecr get-login-password --region ${AWS_DEFAULT_REGION} | \
    docker login --username AWS \
    --password-stdin ${REMOTE_URI}/${REPOSITORY}

docker tag ${LOCAL_IMAGE_NAME} ${REMOTE_IMAGE}
docker push ${REMOTE_IMAGE}

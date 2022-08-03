#!/usr/bin/env bash

echo "checking environment.yaml file..."
NEW_ENVIRONMENT=$(conda env export --no-builds | Out-File environment.yaml -Encoding utf8)

if [ -f environment.yaml ]; then
    echo "environment.yaml exists!"
else
    echo "FAILURE: environment.yaml does not exist!"
    conda env export --no-builds | Out-File environment.yaml -Encoding utf8
    exit 1
fi

ENVIRONMENT=$(cat environment.yaml)

if [ "$NEW_ENVIRONMENT" = "$ENVIRONMENT" ]; then
    echo "environment.yaml is up to date!"
    exit 0
else
    echo "FAILURE: environment.yaml is not up to date!"
    conda env export > environment.yaml --no-builds
    exit 1
fi

echo "checking requirements.txt file..."
NEW_REQUIREMENTS=$(pip list --format=freeze)

if [ -f requirements.txt ]; then
    echo "requirements.txt exists!"
else
    echo "FAILURE: requirements.txt does not exist!"
    pip list --format=freeze > requirements.txt
    exit 1
fi

REQUIREMENTS=$(cat requirements.txt)

if [ "$NEW_REQUIREMENTS" = "$REQUIREMENTS" ]; then
    echo "requirements.txt is up to date!"
    exit 0
else
    echo "FAILURE: requirements.txt is not up to date!"
    pip list --format=freeze > requirements.txt
    exit 1
fi

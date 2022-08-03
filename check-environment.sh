#!/usr/bin/env bash

echo "checking environment.yaml file..."

ENVIRONMENT=$(cat environment.yaml)

CONDA=`/mnt/c/Users/adamgifford_behavr.DESKTOP-PQU0D8M/anaconda3/Scripts/conda`

NEW_ENVIRONMENT=$($CONDA env export > environment.yaml --no-builds)

NEW_REQUIREMENTS=$(pip list --format=freeze)

REQUIREMENTS=$(cat requirements.txt)

echo "checking environment.yaml file..."

if [ "$NEW_ENVIRONMENT" = "$ENVIRONMENT" ]; then
    echo "environment.yaml is up to date!"
    exit 0
else
    echo "FAILURE: environment.yaml is not up to date!"
    $CONDA env export > environment.yaml --no-builds
    exit 1
fi

if [ -f environment.yaml ]; then
    echo "environment.yaml exists!"
else
    echo "FAILURE: environment.yaml does not exist!"
    conda env export --no-builds
    exit 1
fi

echo "checking requirements.txt file..."

if [ "$NEW_REQUIREMENTS" = "$REQUIREMENTS" ]; then
    echo "requirements.txt is up to date!"
    exit 0
else
    echo "FAILURE: requirements.txt is not up to date!"
    pip list --format=freeze > requirements.txt
    exit 1
fi

if [ -f requirements.txt ]; then
    echo "requirements.txt exists!"
else
    echo "FAILURE: requirements.txt does not exist!"
    pip list --format=freeze > requirements.txt
    echo "-e .\n" >> requirements.txt
    exit 1
fi

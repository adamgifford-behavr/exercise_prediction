# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

ARG MODEL_LOCATION

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY app.py .
# will copy all the time, but not used if S3 bucket and credentials are provided in
# docker-compose
COPY models models

ENV FLASK_APP=app.py
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=9696"]
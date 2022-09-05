exercise_prediction
==============================

This project aims to utilize accelerometer and gyroscope data from wearable
sensors to predict exercise categories. It uses the [Exercise Recognition from
Wearable Sensors](https://msropendata.com/datasets/799c1167-2c8f-44c4-929c-227bf04e2b9a)
dataset from the Microsoft Research Open Data Repository. Specifically,
it converts the raw x, y, and z measurement data from the acceleromter and
gyroscope into a set of frequency features that are then used to build
a GradientBoostingClassifier to classify the activity into different
exercise categories (as defined in tags included in the dataset). The end
goal of this project is the ability to simulate streaming sensor data
and generate "live" predictions of the exercise activity.

Project Organization
------------

    ├── LICENSE
    ├── Makefile            <- Makefile with commands like `make data` or `make featrues`
    ├── pyproject.toml      <- Settings for linting and formating
    ├── README.md           <- The top-level README for developers using this project.
    ├── setup.py            <- makes project pip installable (pip install -e .) so src can
    |                          be imported
    ├── test_environment.py <- Script to test base python environment
    ├── data
    │   ├── external        <- Data from third party sources (empty).
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final, canonical data set for modeling (empty, final
    |   |                      store in database).
    │   └── raw             <- The original, immutable data dump.
    │
    ├── docs                <- Detailed Sphinx documentation
    │
    ├── models              <- Best trained and serialized GradientBoostingClassifier model.
    │
    ├── notebooks           <- Jupyter notebooks for setup, exploration, code templating,
    |   |                      and debugging. Naming convention is a number (for ordering),
    │   |                      the creator's initials, and a short `-` delimited
    |   |                      description, e.g. `1.0-jqp-initial-data-exploration`.
    │   ├── 0-setup         <- Notebooks for basic setup
    |   ├── 1-exploratory   <- Notebooks for data exploration
    |   ├── 2-modeling      <- Notebooks for model exploration
    |   └── 3-reports       <- Notebooks for reports (empty)
    ├── references          <- Referene material for the dataset.
    │
    ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc (empty).
    │   └── figures         <- Generated graphics and figures to be used in reporting
    |                          (empty)
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis
    |                          environment
    │
    ├── src                 <- Source code for use in this project.
    │   │
    │   ├── data            <- Scripts to download or generate data
    │   │   ├── make_dataset.py         <- main code for preprocessing data
    |   |   ├── activities.json         <- contains all activity labels
    |   |   ├── activity_groupings.json <- useful label groupings for simplified
    |   |   |                              classification
    |   |   ├── test_split_crit.json    <- criteria for splitting data files into train/val
    |   |   |                              and test datasets
    |   |   └── val_split_crit.json     <- criteria for splitting train/val data files into
    |   |                                  train vs. val
    │   │
    │   ├── features        <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py          <- main code for building features table
    |   |   ├── datafile_group_splits.json <- lists of files organized by group (i.e.,
    |   |   |                                 training, validation, testing, simulation)
    |   |   ├── features_tables.py         <- SQLAlchemy table object definitions for
    |   |   |                                 features tables
    |   |   ├── frequency_features.json    <- signal frequencies to extract for each raw
    |   |   |                                 signal measurement
    |   |   └── metaparams.json            <- metaparameters that define the featurization
    |   |                                     pipeline
    │   │
    │   ├── models          <- Scripts to train models and then use trained models to make
    │   │   │                  predictions
    │   │   ├── score_batch.py          <- Main script for batch scoring
    │   │   ├── train_model.py          <- Main script for model training
    |   |   ├── initial_points_gbc.json <- Starting point for GradientBoostingClassifier
    |   |   |                              hyperparameter search (greatly speeds up fit)
    |   |   ├── model_search.json       <- Data that define the classifier and search
    |   |   |                              parameters
    |   |   └── predictions_tables.py   <- SQLAlchemy table object definitions for recording
    |   |                                  predictions
    │   │
    |   ├── monitor        <- Scripts to simulate streaming prediction monitoring
    |   |   ├── evidently service  <- directory containing monitoring service setup
    |   |   |   ├── config           <- configurations directory
    |   |   |   |   ├── grafana_dashboards.yaml  <- configuration for Grafana dashboards
    |   |   |   |   ├── grafana_datasources.yaml <- configuration for Grafana datasources
    |   |   |   |   ├── prometheus.yaml          <- configuration for Prometheus database
    |   |   |   ├── dashboards       <- dashboard JSON directory
    |   |   |   |   ├── cat_target_drift.json <- configuration for categorical target drift
    |   |   |   |   |                            monitoring
    |   |   |   |   ├── data_drift.json       <- configuration for data drift monitoring
    |   |   |   ├── datasets         <- reference datasets folder (empty to start)
    |   |   |   ├── app.py           <- Flask app for monitoring
    |   |   |   ├── congig.yaml      <- Evidently configurations
    |   |   |   ├── Dockerfile       <- setup for monitoring service container
    |   |   |   ├── requirements.txt <- requirements for monitoring service
    |   |   ├── prediction service <- directory containing prediction service setup
    |   |   |   ├── Dockerfile       <- setup for prediction service container
    |   |   |   ├── requirements.txt <- requirements for the prediction service
    |   |   ├── docker-compose     <- file to compose monitoring and prediction services
    |   |   ├── prepare.py         <- setup script to prepare to simulate monitoring
    |   |   ├── requirements.txt   <- requirements for streaming simulation
    |   |   └── send_data.py       <- script that simulates data streaming
    |   |
    │   └── orchestration   <- Scripts to set up training and scoring orchestration with
    |       |                  Prefect
    │       ├── deploy_score_batch.py      <- code to set up batch scoring deployment
    |       ├── deploy_train.py            <- code to set up training deployment
    |       ├── initial_points_gbc.json    <- Starting point for GradientBoostingClassifier
    |       |                                 hyperparameter search (greatly speeds up fit)
    |       ├── model_search.json          <- Data that define the classifier and search
    |       |                                 parameters
    |       ├── orchestrate_score_batch.py <- code to set up batch scoring as a Prefect flow
    |       ├── orchestrate_train.py       <- code to set up training as a Prefect flow
    |       ├── score_flow-deploynent.yaml <- config for deployment of batch scoring flow
    |       └── train_flow-deployment.yaml <- config for deployment of training flow
    │
    └── tests             <- Scripts to run unit testing

Getting Started
------------
For more detailed instructions, see the `docs`.

**Installation**
1. Clone the repo
```
(base) $ git clone https://github.com/adamgifford-behavr/exercise_prediction.git
```
2. Download the dataset from the Microsoft Research Open Data Repository
[here](https://msropendata.com/datasets/799c1167-2c8f-44c4-929c-227bf04e2b9a).
3. Create a virtual environment
```
conda create --name <env_name> --file requirements.txt
```
4. Create and a virtual environment
```
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```
5. Either set up a PostgreSQL database called "feature_store" in AWS or download locally
and create a password for the default "postgres" user. For local setup, the URI for the
database defaults to localhost:5432. For cloud deployment, find the endpoint connection
string after setup. Store the URI/endpoint and password in a .env file in the parent
directory as:

```
FEATURE_STORE_URI=<URI or ENDPOINT HERE>
FEATURE_STORE_PW=<password here>
```

Similarly, you will need to create a database for MLflow called "mlflow_backend_db" and
an owner for the database called "mlflow". Store the URI/endpoint and password for the
database in .env:

```
MLFLOW_DB_URI=<URI or ENDPOINT HERE>
```

**Data Processing**

To process the data, simply run:
```
(exercise_prediction) $ cd src/data
(exercise_prediction) $ python make_dataset.py
```

**Featurization**

To build features for modeling, run:
```
(exercise_prediction) $ cd src/features
(exercise_prediction) $ python build_features.py
```

**Model Training**

For stand-alone training:
```
(exercise_prediction) $ mlflow server \
    --backend-store-uri postgresql://mlflow:MLFLOW_DB_PW@MLFLOW_DB_URI/mlflow_backend_db \
    --default-artifact-root ROOT_DIRECTORY
(exercise_prediction) $ cd src/models
(exercise_prediction) $ python train_model.py
```

For orchestrated training:
```
(exercise_prediction) $ mlflow server \
    --backend-store-uri postgresql://mlflow:MLFLOW_DB_PW@MLFLOW_DB_URI/mlflow_backend_db \
    --default-artifact-root ROOT_DIRECTORY
(exercise_prediction) $ prefect orion start
(exercise_prediction) $ cd src/orchestration
(exercise_prediction) $ prefect deployment build \
    orchestrate_train.py:train_flow \
    -n 'Main Model-Training Flow' \
    -q 'scheduled_training_flow'
(exercise_prediction) $ prefect deployment apply train_flow-deployment.yaml
(exercise_prediction) $ prefect agent start -q 'scheduled_training_flow'
```

**Model Scoring**

For stand-alone batch scoring:
```
(exercise_prediction) $ cd src/models/
(exercise_prediction) $ pythonscore_batch.py
```

For orchestrated batch scoring:
```
(exercise_prediction) $ cd src/orchestration
(exercise_prediction) $ prefect deployment build \
    orchestrate_score_batch.py:score_flow \
    -n 'Main Model-Scoring Flow' \
    -q 'scheduled_scoring_flow'
(exercise_prediction) $ prefect deployment apply score_flow-deployment.yaml
(exercise_prediction) $ prefect agent start -q 'scheduled_scoring_flow'
```

**Model Monitoring**

To simulate a streaming monitoring service:
```
(exercise_prediction) $ cp models/model.pkl src/monitor/
(exercise_prediction) $ cd src/monitor
(exercise_prediction) $ python prepare.py
(exercise_prediction) $ docker-compose up
(exercise_prediction) $ python src/monitor/send_data.py
```
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

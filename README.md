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

## Project Organization

    ├── LICENSE
    ├── Makefile                <- Makefile with commands like `make data` or `make featrues`
    ├── pyproject.toml          <- Settings for linting and formating
    ├── README.md               <- The top-level README for developers using this project.
    ├── setup.py                <- makes project pip installable (pip install -e .) so src can
    |                              be imported
    ├── test_environment.py     <- Script to test base python environment
    ├── .pre-commit-config.yaml <- configuration for pre-commit hooks
    ├── data                    <- main local data folder
    │   ├── external            <- Data from third party sources (empty).
    │   ├── interim             <- Intermediate data that has been transformed.
    │   ├── processed           <- The final, canonical data set for modeling (empty, final
    |   |                          store in database).
    │   └── raw                 <- The original, immutable data dump.
    │
    ├── docs                        <- Detailed Sphinx documentation
    |   ├── html                    <- html files of detailed documentation
    |   ├── commands.rst            <- raw commands file before build
    |   ├── conf.py                 <- Sphinx configuration file
    |   ├── getting-started.rst     <- raw main documentation file before build
    |   ├── index.rst               <- raw index file before build
    |   ├── make.bat                <- batch file for make commands on Windows
    |   └── Makefile                <- to build documentation in different formats
    |
    ├── infrastructure          <- code for infrastructure resource setup
    |   ├── modules             <- code for individual infrastructure resources
    |   |   ├── ecr             <- code for ecr resourse setup
    |   |   |   ├── main.tf         <- main Terraform code for ecr setup
    |   |   |   └── variables.tf    <- variables for ecr setup
    |   |   ├── kinesis         <- code for kinesis resourse setup
    |   |   |   ├── main.tf         <- main Terraform code for kinesis setup
    |   |   |   └── variables.tf    <- variables for kinesis setup
    |   |   ├── lambda          <- code for lambda resourse setup
    |   |   |   ├── iam.tf          <- permissions setup for lambda
    |   |   |   ├── main.tf         <- main Terraform code for lambda setup
    |   |   |   └── variables.tf    <- variables for lambda setup
    |   |   └──s3               <- code for s3 resourse setup
    |   |      ├── main.tf          <- main Terraform code for s3 setup
    |   |      └── variables.tf     <- variables for s3 setup
    |   ├── vars                <- directory to store variables folder (empty)
    |   ├── main.tf             <- main Terraform setup code
    |   └── variables.tf        <- main variables code
    |
    ├── integration_tests       <- directory for integration tests with Docker and kinesis
    |   ├── data.b64            <- encoded data for testing purposes
    |   ├── docker-compose.yaml <- file to compose integration testing with localstack
    |   ├── run.sh              <- main script to run integration tests
    |   ├── test_docker.py      <- script to test posting data to Docker container to
    |   |                          receive prediction
    |   └── test_kinesis.py     <- script to test posting predictions to kinesis stream
    │
    ├── models                  <- Best trained and serialized GradientBoostingClassifier
    |   |                         model.
    |   ├── conda.yaml          <- requirements for conda environment setup
    |   ├── MLmodel             <- MLflow model metadata file
    |   ├── model.pkl           <- serialized model
    |   ├── python_env.yaml     <- environemnt yaml
    |   └── requirements.txt    <- requirements as text
    ├── notebooks               <- Jupyter notebooks for setup, exploration, code templating,
    |   |                          and debugging. Naming convention is a number (for ordering),
    │   |                          the creator's initials, and a short `-` delimited
    |   |                          description, e.g. `1.0-jqp-initial-data-exploration`.
    │   ├── 0-setup             <- Notebooks for basic setup
    |   ├── 1-exploratory       <- Notebooks for data exploration
    |   ├── 2-modeling          <- Notebooks for model exploration
    |   └── 3-reports           <- Notebooks for reports (empty)
    ├── references              <- Referene material for the dataset.
    ├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc (empty).
    │   └── figures             <- Generated graphics and figures to be used in reporting
    |                              (empty)
    ├── requirements.txt        <- The requirements file for reproducing the analysis
    |                              environment
    ├── scripts                 <- bash scripts for deployments/testing
    |   ├── deploy_manual.sh        <- script to mock manually deploying staging to prod
    |   ├── publish.sh              <- script to publish Docker image to ECR
    |   └── test_cloud_e2e.sh       <- script to test Terraform resource setup end to end
    ├── src                     <- Source code for use in this project.
    │   ├── data                <- Scripts to download or generate data
    │   │   ├── make_dataset.py         <- main code for preprocessing data
    |   |   ├── activities.json         <- contains all activity labels
    |   |   ├── activity_groupings.json <- useful label groupings for simplified
    |   |   |                              classification
    |   |   ├── test_split_crit.json    <- criteria for splitting data files into train/val
    |   |   |                              and test datasets
    |   |   └── val_split_crit.json     <- criteria for splitting train/val data files into
    |   |                                  train vs. val
    |   ├── deployment          <- Scripts to deploy the model in batch and
    |   |   |                      streaming modes
    |   |   ├── batch           <- scripts to deploy in batch mode
    |   |   |   ├── deploy_score_batch.py      <- code to deploy the orchestration
    |   |   |   └── orchestrate_score_batch.py <- main code for setting up scoring
    |   |   |                                     orchestration in batch mode
    |   |   ├── streaming       <- scripts to deploy in streaming mode
    |   |   |   ├── Dockerfile                 <- build file for streaming service
    |   |   |   ├── frequency_features.json    <- signal frequencies to extract for each raw
    |   |   |   |                                 signal measurement
    |   |   |   ├── lambda_function.py         <- lambda function handler for use with AWS
    |   |   |   |                                 lambda
    |   |   |   ├── model.py                   <- main model prediction service module
    |   |   |   ├── Pipfile                    <- minimal environment setup
    |   |   |   ├── Pipfile.lock               <- minimal environment setup
    |   |   |   ├── signals_stream.py          <- setup script to test kinesis functionality
    |   |   |   └── test_docker.py             <- script to test docker container prediction
    |   |   |                                     service
    |   |   └── web_service     <- scripts to deploy as web service
    |   |       ├── Dockerfile
    |   |       ├── frequency_features.json                    <- signal frequencies to
    |   |       |                                                 extract for each raw
    |   |       |                                                 signal measurement
    |   |       ├── Pipfile                                    <- minimal environment setup
    |   |       ├── Pipfile.lock                               <- minimal environment setup
    |   |       ├── predict.py                                 <- main model prediction
    |   |       |                                                 app
    |   |       ├── sample_fileID134_subjID512_dataID1.json    <- sample data to test
    |   |       |                                                 web service
    |   |       └── test.py
    │   ├── features            <- Scripts to turn raw data into features for modeling
    │   │   ├── build_features.py          <- main code for building features table
    |   |   ├── features_tables.py         <- SQLAlchemy table object definitions for
    |   |   |                                 features tables
    |   |   ├── frequency_features.json    <- signal frequencies to extract for each raw
    |   |   |                                 signal measurement
    |   |   └── metaparams.json            <- metaparameters that define the featurization
    |   |                                     pipeline
    |   ├── infrastructure        <- Terraform code for resource managament
    |   |   ├── modules               <- directory for code defining resources
    |   |   ├── vars                  <- directory for code defining resource variables
    |   |   ├── main.tf           <- main Terraform script
    |   |   └── variables.tf      <- variables for main Terraform script
    │   ├── models              <- Scripts to train models and then use trained models to make
    │   │   │                      predictions in stand-alone mode
    │   │   ├── score_batch.py              <- Main script for batch scoring
    │   │   ├── train_model.py              <- Main script for model training
    |   |   ├── initial_points_gbc.json     <- Starting point for GradientBoostingClassifier
    |   |   |                                  hyperparameter search (greatly speeds up fit)
    |   |   ├── model_search.json           <- Data that define the classifier and search
    |   |   |                                  parameters
    |   |   └── predictions_tables.py       <- SQLAlchemy table object definitions for recording
    |   |                                      predictions
    |   ├── monitor            <- Scripts to simulate streaming prediction monitoring
    |   |   ├── evidently service      <- directory containing monitoring service setup
    |   |   |   ├── config               <- configurations directory
    |   |   |   |   ├── grafana_dashboards.yaml      <- configuration for Grafana dashboards
    |   |   |   |   ├── grafana_datasources.yaml     <- configuration for Grafana datasources
    |   |   |   |   ├── prometheus.yaml              <- configuration for Prometheus database
    |   |   |   ├── dashboards           <- dashboard JSON directory
    |   |   |   |   ├── cat_target_drift.json              <- configuration for
    |   |   |   |   |                                         categorical target drift
    |   |   |   |   |                                         monitoring
    |   |   |   |   ├── classification_performance.json    <- configuration for data
    |   |   |   |   |                                         classification performance
    |   |   |   |   |                                         monitoring
    |   |   |   |   └── data_drift.json                    <- configuration for data
    |   |   |   |                                             drift monitoring
    |   |   |   ├── datasets             <- reference datasets folder (empty to start)
    |   |   |   ├── app.py               <- Flask app for monitoring
    |   |   |   ├── congig.yaml          <- Evidently configurations
    |   |   |   ├── Dockerfile           <- setup for monitoring service container
    |   |   |   ├── requirements.txt     <- requirements for monitoring service
    |   |   ├── prediction service       <- directory containing prediction service setup
    |   |   |   ├── Dockerfile           <- setup for prediction service container
    |   |   |   ├── requirements.txt     <- requirements for the prediction service
    |   |   ├── docker-compose           <- file to compose monitoring and prediction services
    |   |   ├── prepare.py               <- setup script to prepare to simulate monitoring
    |   |   ├── requirements.txt         <- requirements for streaming simulation
    |   |   └── send_data.py             <- script that simulates data streaming
    │   └── orchestration       <- Scripts to set up orchestrated training
    |       ├── deploy_train.py            <- code to set up training deployment
    |       ├── initial_points_gbc.json    <- Starting point for GradientBoostingClassifier
    |       |                                 hyperparameter search (greatly speeds up fit)
    |       ├── model_search.json          <- Data that define the classifier and search
    |       |                                 parameters
    |       └── orchestrate_train.py       <- code to set up training as a Prefect flow
    └── tests             <- Scripts to run unit testing
        ├── example_decoded_data.json       <- example decoded data for testing model
        |                                      deployment
        ├── example_features_sample.json    <- example features data for testing model
        |                                      deployment
        ├── example_frequency_features.json <- example frequency features data for testing
        |                                      model deployment
        ├── example_prepared_data.p         <- example prepared features data for testing
        |                                      model deployment
        ├── expected_features_df_data.json  <- data to test main featurization code and
        |                                      data preprocessing
        ├── frequency_features.json         <- frequency features needed for testing model
        |                                      deployment
        ├── test_build_features.py          <- script to test functions in
        |                                    ``build_features.py``
        ├── test_make_dataset.py               <- script to test functions in
        |                                    ``make_dataset.py``
        ├── test_model_deployment.py        <- script to test functions in
        |                                    ``src.deployment.streaming.model``
        ├── test_score_batch.py             <- script to test functions in
        |                                    ``score_batch.py``
        └── test_train_model.py              <- script to test functions in
                                              ``train_model.py``

## Getting Started
For more detailed instructions (including ``make`` commands, testing & formatting, and infrastructure), see the `docs`.

### **Installation**
1. Clone the repo
```
$ git clone https://github.com/adamgifford-behavr/exercise_prediction.git
```
2. Download the dataset from the Microsoft Research Open Data Repository
[here](https://msropendata.com/datasets/799c1167-2c8f-44c4-929c-227bf04e2b9a).
3. Create a and activate a virtual environment
```
$ conda create --name <env_name> python=3.10
$ conda activate <env_name>
```
4. Install requirements
```
$ python -m pip install -U pip setuptools wheel
$ python -m pip install -r requirements.txt
$ pre-commit install
```
5.  Either set up a PostgreSQL database called AWS or download locally, create a
password for the default "postgres" user, and an intial database. For local setup, the
URI for the database defaults to ``localhost:5432/<db_name>``, where ``db_name`` is the
name of your database. For cloud deployment, find the endpoint connection string after
setup, which will be something like
``<rds_instance>.XXXXXXXXXXXX.us-east-1.rds.amazonaws.com/<db_name>``, where
``rds_instance`` is the name of the database instace you created. Store the
URI/endpoint and password in a .env file in the parent directory as:

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

### *Optional*
#### *Setting up `asw-cli`*
1. See instructions on setting up the `aws-cli` [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-prereqs.html).

2. Configure `aws-cli`

```
$ aws configure
AWS Access Key ID [None]: xxx
AWS Secret Access Key [None]: xxx
Default region name [None]: xxx
Default output format [None]:
```

3. Verify aws config

```
$ aws sts get-caller-identity
```

#### *Syncing data to S3*
If you would like to be able to sync your data to an S3 bucket, you will need to also set
the following environment variables:

```
$ export S3_BUCKET=<s3_bucket_name/>
$ export AWS_PROFILE=<name_of_config_profile>
```

where ``name_of_config_profile`` is the name of your AWS profile in "~/.aws/config" (
typically `default` by default).

#### *Cloud resource management with Terraform*
To set up the AWS infrastructure to serve the streaming prediction service, you first will need to make some changes to the main Terraform file:

- *infrastructure/main.tf*

```terraform
terraform {
  required_version = ">= 1.0"
  backend "s3" {
    bucket  = XXX # pre-existing bucket name goes here
    key     = "exercise-prediction-stg.tfstate"
    region  = XXX # your default AWS region
    encrypt = true
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}
```

Additionally, you should create the following files:

- *infrastructure/vars/stg.tfvars*

```terraform
source_stream_name         = "stg_signals_stream"
output_stream_name         = "stg_predictions_stream"
model_bucket               = XXX # the desired staging model bucket goes here
lambda_function_local_path = "../src/deployment/streaming/lambda_function.py"
docker_image_local_path    = "../src/deployment/streaming/Dockerfile"
ecr_repo_name              = XXX # the desired staging ECR repo goes here
lambda_function_name       = XXX # the desired staging lambda function goes here
```

- *infrastructure/vars/prod.tfvars*

```terraform
source_stream_name         = "prod_signals_stream"
output_stream_name         = "prod_predictions_stream"
model_bucket               = XXX # the desired production model bucket goes here
lambda_function_local_path = "../src/deployment/streaming/lambda_function.py"
docker_image_local_path    = "../src/deployment/streaming/Dockerfile"
ecr_repo_name              = XXX # the desired production ECR repo goes here
lambda_function_name       = XXX # the desired production lambda function goes here
```

Once these steps are completed, perform the following:
```
$ cd infrastructure
$ terraform init
$ terraform apply -var-file=vars/<environment_file>
```
where `environment_file` is either *stg.tfvars* or *prod.tfvars* depending on if you are setting up a staging or production environment.


**Data Processing**

To process the data, simply run:
```sh
$ cd src/data
$ python make_dataset.py
```

**Featurization**

To build features for modeling, run:
```sh
$ cd src/features
$ python build_features.py
```

**Model Training**

For orchestrated training, in one terminal:
```sh
$ mlflow server \
    --backend-store-uri postgresql://mlflow:MLFLOW_DB_PW@MLFLOW_DB_URI/mlflow_backend_db \
    --default-artifact-root ROOT_DIRECTORY
```
In a second terminal:
```sh
$ prefect config set \
        PREFECT_ORION_UI_API_URL="http://<EXTERNAL-IP>:4200/api"
$ prefect orion start --host 0.0.0.0
```
In a third terminal:
```sh
$ cd src/orchestration
$ prefect deployment build \
    orchestrate_train.py:train_flow \
    -n 'Main Model-Training Flow' \
    -q 'scheduled_training_flow'
$ prefect deployment apply train_flow-deployment.yaml
$ prefect agent start -q 'scheduled_training_flow'
```

where ``ROOT_DIRECTORY`` is the directory your artifacts will be stored (generally
`mlruns` or a remote storage container like S3) and ``EXTERNAL-IP`` is the address of
your cloud (e.g., AWS EC2) instance.

From there, you can start the training manually from the Prefect UI at `http://<EXTERNAL-IP>:4200`.

**Model Deployment**

*For orchestrated batch scoring*:
```sh
$ cd src/deployment/batch
$ prefect deployment build \
	orchestrate_score_batch.py:score_flow \
	-n 'Main Model-Scoring Flow' \
	-q 'manual_scoring_flow'
$ prefect deployment apply score_flow-deployment.yaml
$ prefect agent start -q 'manual_scoring_flow'
```

From there, you can start the scoring manually from the Prefect UI at `http://<EXTERNAL-IP>:4200`.

*For web-service deployment*:<br>
In one terminal:

```sh
$ cp -R models src/deployment/web_service
$ cd src/deployment/web_service
$ docker build -t exercise-prediction-webservice:v1 .
$ docker run -itd --rm -p 9696:9696 exercise-prediction-webservice:v1
$ python test.py
```

The Docker container is run in detached mode, use ``docker stop ...`` to stop the container
once finished testing. See ``docker stop --help`` for details.

*For streaming deployment*:

```sh
(exercise_prediction) $ cp -R models src/deployment/streaming
(exercise_prediction) $ cd src/deployment/streaming
(exercise_prediction) $ docker build -t exercise-prediction-streaming:v1 .
(exercise_prediction) $ docker run -itd --rm -p 8080:8080 exercise-prediction-streaming:v1
(exercise_prediction) $ python test_docker.py
```

The Docker container is run in detached mode, use ``docker stop ...`` to stop the container
once finished testing. See ``docker stop --help`` for details.

**Publishing**

For information on publishing the streaming service container image, see the detailed
documentation in `docs`.

**Model Monitoring**

To simulate a streaming monitoring service:

```sh
$ cp models -r src/monitor/prediction_service
$ cd src/monitor
$ python prepare.py
$ docker-compose up
```
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Getting started
===============

Problem description
-------------------

This project aims to utilize accelerometer and gyroscope data from wearable
sensors to predict exercise categories. It uses the `"Exercise Recognition from
Wearable Sensors"
<https://msropendata.com/datasets/799c1167-2c8f-44c4-929c-227bf04e2b9a>`_
dataset from the Microsoft Research Open Data Repository. Specifically,
it converts the raw x, y, and z measurement data from the acceleromter and
gyroscope into a set of frequency features that are then used to build
a GradientBoostingClassifier to classify the activity into different
exercise categories (as defined in tags included in the dataset). The end
goal of this project is the ability to simulate streaming sensor data
and generate "live" predictions of the exercise activity.

.. _installation:

Installation
------------

1. Clone the directory from `my GitHub page
<https://github.com/adamgifford-behavr/exercise_prediction.git>`_.

.. code-block:: console

    (base) $ git clone https://github.com/adamgifford-behavr/exercise_prediction.git

2. Download the dataset from the Microsoft Research Open Data Repository
`here <https://msropendata.com/datasets/799c1167-2c8f-44c4-929c-227bf04e2b9a>`_
(you will need to login or create an account and agree to the terms of use in
order to download the data, but it is free). Place the contents in the
*data/raw/* directory.

3. Create a virtual environment for the entire project using your favorite method. For example,
using make:

.. code-block:: console

    (base) $ make create_environment

or, using conda:

.. code-block:: console

    (base) $ conda create --name <env_name> python=3.10

where ``env_name`` is whatever name you want to provide for the environment. If using
``make``, ``env_name`` defaults to "exercise_prediction".

4. Activate your environment with either ``conda activate <env_name>`` or
``source activate <env_name>``, then install requirements using make:

.. code-block:: console

    (env_name) $ make requirements

or, using pip:

.. code-block:: console

    (env_name) $ python -m pip install -U pip setuptools wheel
    (env_name) $ python -m pip install -r requirements.txt
    (env_name) $ pre-commit install

where ``env_name`` is whatever name you want to provide for the environment. If using
``make``, ``env_name`` defaults to "exercise_prediction".

5. Either set up a PostgreSQL database called AWS or download locally, create a
password for the default "postgres" user, and an intial database. For local setup, the
URI for the database defaults to ``localhost:5432/<db_name>``, where ``db_name`` is the
name of your database. For cloud deployment, find the endpoint connection string after
setup, which will be something like
``<rds_instance>.XXXXXXXXXXXX.us-east-1.rds.amazonaws.com/<db_name>``, where
``rds_instance`` is the name of the database instace you created. Store the
URI/endpoint and password in a .env file in the parent directory as:

.. code-block:: text

    FEATURE_STORE_URI=<URI or ENDPOINT HERE>
    FEATURE_STORE_PW=<password here>

Similarly, you will need to create a database for MLflow called "mlflow_db" and
an owner for the database called "mlflow". Store the URI/endpoint and password for the
database in .env:

.. code-block:: text

    MLFLOW_DB_URI=<URI or ENDPOINT HERE>
    MLFLOW_DB_PW=<password here>

.. note::

    You don't need ``MLFLOW_DB_PW`` for any code in the package, but you will need it
    to start the MLflow server.


Optional
~~~~~~~~

Setting up ``asw-cli``
^^^^^^^^^^^^^^^^^^^^^^

To set up the ``aws-cli``, follow these instructions:

1. See instructions on setting up the ``aws-cli`` `here`_.

.. _a link: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-prereqs.html

2. Configure ``aws-cli``

.. code-block:: console

    $ aws configure
    AWS Access Key ID [None]: xxx
    AWS Secret Access Key [None]: xxx
    Default region name [None]: xxx
    Default output format [None]:

3. Verify aws config

.. code-block:: console

    $ aws sts get-caller-identity

Syncing data to S3
^^^^^^^^^^^^^^^^^^

If you would like to be able to sync your data to an S3 bucket, you will need to also set
the following environment variables:

.. code-block:: console

    (exercise_prediction) $ export S3_BUCKET=<s3_bucket_name/>
    (exercise_prediction) $ export AWS_PROFILE=<name_of_config_profile>

where ``name_of_config_profile`` is the name of your AWS profile in *~/.aws/config* (
typically *default* by default).

.. _Infrastructure:

Cloud resource management with Terraform
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To set up the AWS infrastructure to serve the streaming prediction service, you first
will need to make some changes to the main Terraform file:

.. code-block:: terraform
    :caption: *infrastructure/main.tf*

    terraform {
    required_version = ">= 1.0"
    backend "s3" {
        bucket  = XXX # pre-existing bucket name goes here
        key     = "exercise-prediction-stg.tfstate"
        region  = YYY # your default AWS region
        encrypt = true
    }
    required_providers {
        aws = {
        source  = "hashicorp/aws"
        version = "~> 4.0"
        }
    }
    }

where ``XXX`` and ``YYY`` should be replaced with your existing S3 bucket and default
region, respectively.

Additionally, you should create the following files:

.. code-block:: terraform
    :caption: *infrastructure/vars/stg.tfvars*

    source_stream_name         = "stg_signals_stream"
    output_stream_name         = "stg_predictions_stream"
    model_bucket               = AAA # the desired staging model bucket goes here
    lambda_function_local_path = "../src/deployment/streaming/lambda_function.py"
    docker_image_local_path    = "../src/deployment/streaming/Dockerfile"
    ecr_repo_name              = BBB # the desired staging ECR repo goes here
    lambda_function_name       = CCC # the desired staging lambda function goes here

where ``AAA``, ``BBB``, and ``CCC`` are the names of the S3 bucket where your model is
located, the repository to store your docker contianer, and the name of your lambda
function, respectively (and ideally, each name should start with *stg_* to signify these
are resources in the staging environment).

.. code-block:: terraform
    :caption: *infrastructure/vars/prod.tfvars*

    source_stream_name         = "prod_signals_stream"
    output_stream_name         = "prod_predictions_stream"
    model_bucket               = DDD # the desired production model bucket goes here
    lambda_function_local_path = "../src/deployment/streaming/lambda_function.py"
    docker_image_local_path    = "../src/deployment/streaming/Dockerfile"
    ecr_repo_name              = EEE # the desired production ECR repo goes here
    lambda_function_name       = FFF # the desired production lambda function goes here

where ``DDD``, ``EEE``, and ``FFF`` are the names of the S3 bucket where your model is
located, the repository to store your docker contianer, and the name of your lambda
function, respectively (and ideally, each name should start with *prod_* to signify these
are resources in the production environment).

Once these steps are completed, you can use either ``make`` or the console to create the
resources. With ``make``, use:

.. code-block:: console

    $ make create_stage_infra

to create resources for the staging environment or use:

.. code-block:: console

    $ make create_prod_infra

to create resources for the production environment.

Alternatively, you can perform the same functions using console commands:

.. code-block:: console

    $ cd infrastructure
    $ terraform init
    $ terraform apply -var-file=vars/<environment_file>

where ``environment_file`` is either *stg.tfvars* or *prod.tfvars* depending on if you
are setting up a staging or production environment.

.. note::

    The infrastructure setup creates resources (such as Amazon Kinesis) that can be costly.
    To avoid excessive usage costs, make sure to delete these resourses as soon as you
    are finished with them. The commands ``make destroy_stage_infra`` and
    ``make destroy_prod_infra`` are the easiest ways to do this.

Data Processing
---------------

Preprocessing the data
~~~~~~~~~~~~~~~~~~~~~~

You first need to load, restructure, and convert the raw data that is in
MATLAB's data format into a series of PARQUET files that will make further
processing easier. We are working solely on the raw data file
"exercise_data.50.0000_multionly", which contains continuous labeled data
across all test subjects for a variety "real-world environment" activities.
The module src/data/make_dataset.py handles this preprocessing and stores
the resulting files in data/interim.

Quickstart
^^^^^^^^^^

To process the data, simply run either:

.. code-block:: console

    (exercise_prediction) $ make data

or

.. code-block:: console

    (exercise_prediction) $ cd src/data
    (exercise_prediction) $ python make_dataset.py

The details
^^^^^^^^^^^

If you would like more reference on how the raw MATLAB files are structured, see
*notebooks/0-setup/0.1-agifford-TestLoadMatFileAndVerify.ipynb*. Running *make_dataset.py*
also produces the file *src/features/datafile_group_splits.json*, which splits each
PARQUET files into one of 4 groups:

- "train": for model training;
- "validation": for model validation and hyperparameter tuning;
- "test": for model testing and comparing among different model flavors;
- "simulate": for simulating "real-world" web-service, streaming and batch model serving.

This file is a necessary input for *src/features/build_features.py*.

Building the features
~~~~~~~~~~~~~~~~~~~~~

Next, we build frequency features from the raw signals to use in our modeling.

Quickstart
^^^^^^^^^^

To run the code, run either:

.. code-block:: console

    (exercise_prediction) $ make features

or

.. code-block:: console

    (exercise_prediction) $ python src/features/build_features.py \
		src/features/frequency_features.json \
		src/features/datafile_group_splits.json \
		src/features/metaparams.json

When features are completed, find the log that identifies the "id" for the run and
store it in .env as:

.. code-block:: text

    FEATURIZE_ID=<id here>

The details
^^^^^^^^^^^

The logic of the analysis is as follows:

- for each file, signals are binned into 3-second windows (see
  *notebooks/1-agifford-TestLoadMatFileAndVerify/1.4-agifford-DetermineAnalysisWindowSize.ipynb*
  for a detailed run-through on the 3-second window rationale)
- in each window, we compute a Fourier transform of the signal after applying a
  Hanning window
- for each signal (e.g., "accel_x", "accel_y", "gyro_z", etc.), we extract the
  magnitude of a select few frequecnies (see
  *notebooks/1-exploratory/1.3-agifford-FindFrequencyPeaksTraining.ipynb* for a detailed
  run-through of my process for determining the particular frequencies of interest and the
  code for storing the data for use in *build_features.py* )
- these frequency features by raw signal are stored in the file
  *src/features/frequency_feature.json*, which is another necessary input to
  *src/features/build_features.py*.

The final necessary input to *build_features.py* is *src/features/metaparams.json*, which
provides details about the process employed to generate the features. This file is manually
created and already provided in the package. Currently, the file looks as follows:

.. code-block:: json

    {
      "n_fft": 151,
      "spectrum_decibel": false,
      "spectrum_frequencies": "naive",
      "spectrum_method": "fft",
      "spectrum_normalized": true,
      "table_name": "naive_frequency_features",
      "window": "hanning"
    }

Only ``n_fft`` is actually used as a parameter in the function (and as such controls the
size of the window in which to analyze the data). The other parameters are used to generate
a unique "featurization_id" for the feature_store database to identify when an identical
run of the featurization process is conducted to decide whether to skip re-running an
identical featurization process if data for it already exists in the database (in a
future update to this project, these (and potentially other) metaparamters would ultimately
be used in the featurization process to actually implement different feature-building
processes).

Model Training
--------------

Model training can be performed in stand-alone mode (i.e., running locally with no
orchestration) or with orchestration via Prefect. The end result is a series of mlflow
runs to identify the best hyperparameters for the classifier and the "best" mode promoted
to the registry and transitioned to "Staging".

.. note::

    The best model is automatically promoted only to "Staging", under the assumption that,
    in a real scenario, there would be a manual gating to promote the model to "Production".
    However, subsequent scoring and monitoring code assumes the model is promoted to
    "Production". When training is complete, you will need to manually transition the
    model to "Production", or modify the subsequent code to search for the model in
    "Staging".

To perform model training, first you need to start the MLflow server:

.. code-block:: console

    (exercise_prediction) $ mlflow server [-h 0.0.0.0 -p 5000] \
        --backend-store-uri postgresql://mlflow:MLFLOW_DB_PW@MLFLOW_DB_URI \
        --default-artifact-root ROOT_DIRECTORY

where ``ROOT_DIRECTORY`` is the directory your artifacts will be stored (generally
*mlruns* or a remote storage container like S3). The arguments ``-h 0.0.0.0 -p 5000``
are optional for if you are deploying the tracking server to the cloud.

.. note::

    The mlflow server command does not import environment variables ``MLFLOW_DB_PW`` and
    ``MLFLOW_DB_URI``, so these will need to be written out in the command above.

You also may need to define the following environment variables in *.env*:

.. code-block:: text

    EXP_NAME=<experiment name here>

where ``EXP_NAME`` is the desired name of the MLflow experiment. This only needs to be
explicitly defined if you'd like to change the name of the experiment. If changed, just
make sure it remains the same for subsequent model scoring and monitoring (see below).

Stand-alone training
~~~~~~~~~~~~~~~~~~~~

Quickstart
^^^^^^^^^^

To run the training in stand-alone mode, run either:

.. code-block:: console

    (exercise_prediction) $ make stand_alone_train

or

.. code-block:: console

    (exercise_prediction) $ python src/models/train_model.py \
		naive_frequency_features \
		label_group \
		src/models/model_search.json

.. note::

    If you are running a local server, make sure your artifact ``ROOT_DIRECTORY``
    is at the same level as you are where you run model training (i.e., *exercise_prediction*
    for ``make`` or *exercise_prediction/src/models* for ``python``). Alternatively, if running
    *train_model.py* from the parent directory, you'll have to include the relative paths
    of the necessary json inputs (see below).

Model training with *train_model.py* requires 3 inputs, and an optional 4th:

.. py:function:: src.models.train_model

   Return a list of random ingredients as strings.

   :param table_name: the name of the table in the database that contains the data
   :type table_name: str
   :param label_col: the name of the column in the data that contains the labels
   :type label_col: str
   :param model_search_json: This is the path to the JSON file that contains the model
     name, fixed parameters, and search parameters. Defaults to ./model_search.json
   :type model_search_json: str
   :param  initial_points_json: This is the path to the JSON file that
     contains starting points for hyperparameter values for fitting procedure (e.g., to
     use values from previous fit to potentially speed up fitting). Defaults to None
   :type initial_points_json: str or None

   :rtype: None

The details
^^^^^^^^^^^

The file identified by *model_search_json* contains the following information:

.. code-block:: json

    {
      "fixed_paramaters": {
        "n_iter_no_change": 50,
        "random_state": 42,
        "tol": 0.001,
        "warm_start": true
    },
      "fmin_rstate": 42,
      "model": "gradientboostingclassifier",
      "search_parameters": [
        "max_depth",
        "learning_rate",
        "n_estimators",
        "subsample",
        "min_samples_split",
        "min_samples_leaf",
        "max_features"
      ],
      "test_limit": null,
      "train_limit": null,
      "unsearched_parameters": [
        "ccp_alpha",
        "max_leaf_nodes",
        "min_impurity_decrease",
        "min_weight_fraction_leaf"
      ],
      "validation_limit": null
    }

- The "model" input defines the flavor of classifier to fit. Currently, only sklearn
  ``ExtraTreesClassifier``, ``GradientBoostingClassifier``, or ``RandomForestClassifier``
  are supported.
- The "train_limit", "validation_limit", and "test_limit" inputs define how many
  from training, validation, and testing to include in the model fitting. ``null`` values
  for any input means "use all samples". Non-``null`` values are simply for testing and
  debugging the code.
- The "fmin_rstate" is the random state for ``hyperopt.fin`` (for reproducibility).
- Next, there is the "search_parameters" input, which is a list of input hyperparameter
  names to the classifier that will be fit with ``hyperopt``. The global variable
  ``ALL_SEARCH_PARAMS`` in *train_model.py* defines the search spaces for all potential
  hyperparameters of interest across the 3 classifier flavors.
- Finally, there is the "fixed_paramaters" input, which is itself a dictionary of
  inputs to the classifier that are to remain fixed throughout the hyperparameter tuning.
- There is also a parameter "unsearched_parameters", which is a list of other potential
  hyperparameters that **could** be fit for the classifier, but are not. This field is simply
  ignored during training.

.. note::

    If you want to convert any "fixed_paramaters" to "search_parameters", you must
    add them to ``ALL_SEARCH_PARAMS`` with a defined ``hyperopt`` search space. Similarly,
    if you want to test a different classifier, the classifier needs to be imported in
    *train_model.py*, it must be added to the dictionary ``classifiers`` in
    ``train_model._get_named_classifier()``, and any additional search parameters must be
    added to ``ALL_SEARCH_PARAMS`` with defined ``hyperopt`` search spaces.

The file identified by ``initial_points_json`` (if not ``None``) is a manually generated
file that contains either a single set or list of initial points to start with for
hyperparameter tuning. This is potentially useful for, e.g., a manual "warm start" of the
model training on the full dataset from a previous run on a sample of data.

.. note::

    The data in ``initial_points_json`` must match all searched parameters that are
    identified in "search_parameters" in ``model_search_json``. Also, for any categorical
    search parameters that require a search space using ``hp.choice()`` (e.g., ``max_features``
    for ``GradientBoostingClassifier``), you need to input the index associated with that
    parameter value defined by ``hp.choice()`` in ``ALL_SEARCH_PARAMS``. For example, to input
    a value of ``max_features = "log2"`` in your classifier during the hyperparameter search,
    you would need to convert this to ``"max_features": 1`` in ``initial_points_json``. For
    the current  best ``GradientBoostingClassifier`` model, the initial points are set as
    follows:

.. code-block:: json

    {
      "learning_rate": 0.054263643103364075,
      "max_depth": 4,
      "max_features": 0,
      "min_samples_leaf": 0.02665082218633991,
      "min_samples_split": 0.062086662821805284,
      "n_estimators": 1900,
      "subsample": 1.0
    }

If you want to use an ``initial_points_json`` file when you run the code, either run it
using python directly or add an extra line to the ``Makefile`` under the
"stand_alone_train" section that points to the path to the json file:

.. code-block:: make

    stand_alone_train: features
        $(PYTHON_INTERPRETER) src/models/train_model.py \
            naive_frequency_features \
            label_group \
            src/models/model_search.json \
            <INITIAL_POINTS_JSON_PATH>

Orchestrated training
~~~~~~~~~~~~~~~~~~~~~

Alternatively, model training can be orchestrated via Prefect.

Quickstart
^^^^^^^^^^

For orchestrated model training, you also need start a Prefect server:

.. code-block:: console

    (exercise_prediction) $ prefect config set \
        PREFECT_ORION_UI_API_URL="http://EXTERNAL-IP:4200/api"
    (exercise_prediction) $ prefect orion start --host 0.0.0.0

where ``EXTERNAL-IP`` is the address of your cloud (e.g., AWS EC2) instance.

Next, you have the option to set up a cloud storage block to log flow run data. Follow
the instructions from `this site <https://docs.prefect.io/tutorials/storage/>`_ if you
would like to use remote storage for the deployment.

If you created a remote storage-block, create the following environment variable:

.. code-block:: console

    (exercise_prediction) $ export PREFECT_TRAIN_SB=<block_type>/<block_name>

where ``block_type`` is the type of remote storage you used (e.g., "s3") and ``block_name``
is the name of the block you created.

Finally, run either of the following commands to create and deploy the orchestration:

.. code-block:: console

    (exercise_prediction) $ make orchestrate_train

or

.. code-block:: console

    (exercise_prediction) $ cd src/orchestration
    (exercise_prediction) $ prefect deployment build \
		orchestrate_train.py:train_flow \
		-n 'Main Model-Training Flow' \
		-q 'manual_training_flow'
	(exercise_prediction) $ prefect deployment apply train_flow-deployment.yaml
    (exercise_prediction) $ prefect agent start -q 'manual_training_flow'

The details
^^^^^^^^^^^

Running orchestrated training simply calls ``orchestrate_train.train_flow()``, which
is a copy of ``train_model.main()`` with Prefect flow and task decorators. As such, it
requires the same input parameters. The first 3 (``table_name``, ``label_col``, and
``model_search_json``) are provided by default in the function. The final optional parameter
(``initial_points_json``) would need to be provided at flow run time.

If you do not provide a remote storage-block location, the commands will default to using
local storage.

.. note::

    The orchestration is not set up to run on a schedule (since there is no incoming new
    data to re-fit). Therefore, you will need to go to the Prefect UI to manually start
    a run of model training.

Stand-Alone Model Serving
-------------------------

Model Serving can be performed in stand-alone mode with batch scoring. The end result is
to test the "Production" model on simulated new data that was preprocessed by
*build_features.py*.

We simulate scoring the model on new (unseen) data in batch mode by loading in data with
the "simulate" ``dataset_group`` from our features table (which was processed in
*build_features.py*). After scoring, we save the predictions and true labels, along
with a link to each row of data in our features table, to a predictions table in our
``feature_store`` database for further analysis.

.. note::

    Model scoring with default parameter settings requires a model in "Production" stage.
    Transitioning a model to "Production" is simulated as a manual step in this project,
    thus you will have to manually promote the best model from *build_features.py*
    in the MLflow model registry from "Staging" to "Production".

`---`
~~~~~

Quickstart
^^^^^^^^^^

To begin model scoring, using either:

.. code-block:: console

    (exercise_prediction) $ make stand_alone_score_batch

or

.. code-block:: console

    (exercise_prediction) $ python src/models/score_batch.py

The details
^^^^^^^^^^^

*score_batch.py* requires the following inputs:

.. py:function:: src.models.batch_score()

   It loads simulated batch data from a table in the database, applies a model to it,
   and writes the predictions to a table in the database

   :param feature_table: the name of the table in the feature store to load the
      data for scoring. Defaults to naive_frequency_features
   :type feature_table: str
   :param prediction_table: the name of the table in the feature store to log the
      predictions. Defaults to naive_frequency_features_predictions
   :type prediction_table: str
   :param label_col: The name of the column in the feature table that contains the label.
      Defaults to label_group
   :type label_col: str
   :param  model_name: the name of the model in the model registry. Defaults to
      exercise_prediction_naive_feats_pipe
   :type model_name: str
   :param  model_stage: the stage of the model in the model registry. Defaults to Production
   :type model_stage: str

   :rtype: None

The code should be able to run with its default parameters. If the ``prediction_table``
doesn't exist in the database, the code can create it.

.. note::

    If there was an error in a previous run of the code that requires you to drop the
    ``prediction_table`` from the database, you will also need to delete the ``Sequence``
    generator used to auto-increment the table's primary key (otherwise you will get an
    error trying to recreate a ``Sequence`` that already exists). You can do this in
    pgAdmin by right-clicking *Databases > feature_store > Schemas > public > Sequences >
    naive_frequency_features_predictions_naive_frequency_features_p...* and selecting
    "Delete/Drop".

Model Serving
-------------

Model serving can also deployed in batch (orchestrated with Prefect), web-service (in a
docker container), and streaming (with AWS Kinesis and Lambda functions) modes.

Batch mode
~~~~~~~~~~

Quickstart
^^^^^^^^^^

For orchestrated model batch scoring, you also have the option to set up a cloud storage
block. If you created a remote storage-block, create the following environment variable:

.. code-block:: console

    (exercise_prediction) $ export PREFECT_SCORE_BATCH_SB=<block_type>/<block_name>

For orchestrated model batch scoring, you can then run either:

.. code-block:: console

    (exercise_prediction) $ make orchestrate_score_batch

or

.. code-block:: console

    (exercise_prediction) $ cd src/deployment/batch
    (exercise_prediction) $ prefect deployment build \
		orchestrate_score_batch.py:score_flow \
		-n 'Main Model-Scoring Flow' \
		-q 'manual_scoring_flow'
	(exercise_prediction) $ prefect deployment apply score_flow-deployment.yaml
    (exercise_prediction) $ prefect agent start -q 'manual_scoring_flow'

The details
^^^^^^^^^^^

Running orchestrated training simply calls ``orchestrate_batch_score.score_flow()``, which
is a copy of ``score_batch.main()`` with Prefect flow and task decorators. As such, it
requires the same input parameters.

.. note::

    The orchestration is not set up to run on a schedule (since there is no continual
    stream of new data to re-fit). Therefore, you will need to go to the Prefect UI to
    manually start a run of model scoring.


Web service
~~~~~~~~~~~

Quickstart
^^^^^^^^^^

If you would like to test the web service with your own model (stored in S3 for example),
you need to create a *.env* file in *src/deployment/web_service/* with the following
environment variables:

.. code-block:: text

    MODEL_LOCATION=<full bucket path to mlflow models folder>
    AWS_ACCESS_KEY_ID=XXXX
    AWS_SECRET_ACCESS_KEY=XXXX
    AWS_DEFAULT_REGION=XXXX

where ``MODEL_LOCATION`` looks something like:
*s3://<YOUR_BUCKET>/<EXP_ID>/<RUN_ID>/artifacts/models/*.

For use as a web service, simply run one of the following commands:

.. code-block:: console

    (exercise_prediction) $ make deploy_web

or

.. code-block:: console

	(exercise_prediction) $ cp -R models src/deployment/web_service
    (exercise_prediction) $ cd src/deployment/web_service
	(exercise_prediction) $ docker build -t exercise-prediction-webservice:v1 .
	(exercise_prediction) $ docker run -itd --rm -p 9696:9696 exercise-prediction-webservice:v1
	(exercise_prediction) $ python test.py

The ``make`` command will automatically run the container with the *.env* file if it
exists. Otherwise it will run the container with the pretrained model in *models/*.
However, if you want to start the service step by step from the console, replace the
``docker run`` command above with the following:

.. code-block:: console

    (exercise_prediction) $ docker run -itd --rm -p 9696:9696 --env-file .env exercise-prediction-webservice:v1

.. note::

    The docker container is run in detached mode. Make sure to run ``docker stop ...``
    to stop the container when you complete testing (see ``docker stop --help`` for
    details).

The details
^^^^^^^^^^^

The web service works by reading in a "packet" of streaming exercise data, performing
the necessary preprocessing steps of:

1. selecting the appropriate data fields
2. performing a Fourier transform of the data
3. selecting the appropriate frequency features
4. passing the features to ``model.predict()`` to get a prediction for the exercise
5. returning the prediction

The data "packets" sent to the prediction service correspond to windows of streaming data
that contain contiguous samples of recordings in order to perform the necessary
featurizations. The *predict.py* app is designed to accept any duration of data (in
theory), but for example purposes the data sent in *test.py* contains 151 contiguous
samples of data, which is the same number used in the original featurization process.

Streaming
~~~~~~~~~

Quickstart
^^^^^^^^^^
If you would like to test the containerized streaming service with your own model
(stored in S3 for example), create a *.env* file in *src/deployment/streaming/* with the
following environment variables:

.. code-block:: text

    MODEL_BUCKET=<s3 bucket name here>
    MLFLOW_EXPERIMENT_ID=<experiment number here>
    RUN_ID=<run id here>

For use as a streaming service, simply run one of the following commands:

.. code-block:: console

    (exercise_prediction) $ make deploy_streaming

or

.. code-block:: console

	(exercise_prediction) $ cp -R models src/deployment/streaming
	(exercise_prediction) $ cd src/deployment/streaming
	(exercise_prediction) $ docker build -t exercise-prediction-streaming:v1 .
	(exercise_prediction) $ docker run -itd --rm -p 8080:8080 exercise-prediction-streaming:v1
	(exercise_prediction) $ python test_docker.py

The ``make`` command will automatically run the container with the *.env* file if it
exists. Otherwise it will run the container with the pretrained model in *models/*.
However, if you want to start the service step by step from the console and use your own
*.env* file, replace the ``docker run`` command above with the following:

.. code-block:: console

    (exercise_prediction) $ docker run -itd --rm -p 8080:8080 --env-file .env exercise-prediction-streaming:v1

.. note::

    The docker container is run in detached mode. Make sure to run ``docker stop ...``
    to stop the container when you complete testing (see ``docker stop --help`` for
    details).

The details
^^^^^^^^^^^

The streaming service works similarly to the web service, except for essentially employing
AWS Lambda as the "web service" rather than a custom service via Flask. The service runs
in "test" mode by default (which does not attempt to put records to a predictions stream).
If you would like to test putting records to a stream you created, make sure the environment
variables ``PREDICTIONS_STREAM_NAME``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
and ``AWS_DEFAULT_REGION`` are appropriately defined in your *.env* file.

Deployment
----------

Publishing to ECR
~~~~~~~~~~~~~~~~~

The containerized streaming service can be published to Amazon ECR.

Quickstart
^^^^^^^^^^

To publish the streaming service to the container registry, you must first set the
following environment variables:

.. code:: console

    $ export AWS_ACCOUNT_ID=XXX
    $ export AWS_DEFAULT_REGION=YYY
    $ export REPOSITORY=ZZZ

where ``XXX``, ``YYY``, and ``ZZZ`` are your acount id number, your AWS region, and the
name of your repository in ECR where you want to publish the container.

Next, simply run:

.. code:: console

    $ make publish

The details
^^^^^^^^^^^

The publish command depends on a series of other steps before the container is actually
published to ECR:

1. ``make quality_checks`` performs code quality checks (e.g., linting and typing)
2. ``make code_tests`` performs unit tests on the codebase
3. ``make build`` builds the container
4. ``make integration_tests`` performs integration tests for the container using localstack

After those steps are completed successfully, ``make publish`` calls *scripts/publish.sh*
to publish the container. See `Testing`_ below for details on the testing structure.

Manual deployment
~~~~~~~~~~~~~~~~~

Quickstart
^^^^^^^^^^

The streaming service is also set up for manual deployment to the staging environment
created by Terraform. First, you will need to edit the following variables in
*scripts/deploy_manual.sh*:

.. code-block:: bash

    AWS_REGION=xxx

    # Dynamically generated by TF
    export MODEL_BUCKET_PROD=xxx
    export PREDICTIONS_STREAM_NAME=xxx
    export LAMBDA_FUNCTION=xxx

    # Model artifacts bucket from your stand-alone runs of modeling
    export MODEL_BUCKET_DEV=xxx

    # Get latest RUN_ID from latest S3 partition.
    # NOT FOR PRODUCTION!
    # In practice, this is generally picked up from your experiment tracking tool such as MLflow or DVC
    export RUN_ID=xxx
    export EXPERIMENT_ID=xxx

Next, simply run:

.. code-block:: console

    $ ./scripts/deploy_manual.sh

The details
^^^^^^^^^^^

The script *deploy_manual.sh* first copies the model folder from the s3 bucket you used
for model training and registration to the staging environment s3 bucket. It then updates
the variables necessary for the lambda function to connect to the model folder in the
staging environment.

.. note::

    The manual deployment depends upon creation of the staging environment resources
    created with Terraform. See :ref:`Cloud resource management with Terraform <Infrastructure>`
    for infrastructure setup instructions.

Monitoring
----------

Model monitoring is performed on simulated streaming data by taking the data records
labeled "simulate" in our feature_store, pinging a **prediction** service every 3 seconds
(i.e., the current feature window size) to generate a model prediction, and finally
pinging the **evidently** service to monitor performance. It requires a build with
``docker-compose`` and a run of *src/monitor/send_data.py* to stream the data to the
**prediction** and **evidently** services.

`---`
~~~~~

Quickstart
^^^^^^^^^^

To start the **evidently** and **prediction** services, run one of the following to
build and start the docker containers:

.. code-block:: console

    (exercise_prediction) $ make docker_monitor

or

.. code-block:: console

    (exercise_prediction) $ cp models -r src/monitor/prediction_service
    (exercise_prediction) $ cd src/monitor
    (exercise_prediction) $ python prepare.py
    (exercise_prediction) $ docker-compose up

Next, in another terminal start sending data to the services:

.. code-block:: console

    (exercise_prediction) $ python src/monitor/send_data.py

The details
^^^^^^^^^^^

The *prepare.py* script loads the simulation data from the database and stores it as
a separate PARQUET file in *src/monitor* and *src/monitor/evidenctly_service/datasets*.
There is an example model included in *src/monitor/prediction_service* in case one wants
to test the monitoring functionality without running through the rest of the pipeline
(i.e., data processing, featurization, model training).

.. note::

    If you would like to test your own model created during your run-through of the
    pipeline, you must manually copy your *models* (or similarly saved model folder)
    from your model registry/artifact store into *src/monitor/prediction_service/*. The
    prediction services uses ``mlflow.pyfunc.load_model()`` under the hood, so the contents
    of the *models* folder should conform to the requirements necessary for MLflow.
    Alternatively, the prediction service is designed to connect to an S3 bucket if you'd
    prefer to load the model from S3. In order to do that, you must fill in the
    environment variables ``MODEL_LOCATION``, ``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, and ``AWS_DEFAULT_REGION`` in
    *src/monitor/docker-compose.yml*, where ``MODEL_LOCATION`` is the full s3 path to
    your *models* folder.

.. _Testing:

Testing
-------

Quality checks
~~~~~~~~~~~~~~

Quality checks include:
- Package import sorting with ``isort``
- Code formatting with ``black``
- Linting with ``pylint``
- Static type checking ``mypy``
- Security checking with ``bandit``

Quickstart
^^^^^^^^^^

To perform the quality checks, simply run either:

.. code-block:: console

    (exercise_prediction) $ make quality_checks

or

.. code-block:: console

	(exercise_prediction) $ isort --line-length=88 src
	(exercise_prediction) $ black --line-length=88 src
	(exercise_prediction) $ pylint -rn -sn --ignore-paths=tests,integration_tests src
	(exercise_prediction) $ mypy --no-strict-optional --ignore-missing-imports --exclude "app.py" --exclude "^tests/" --exclude "^integration_tests/" src
	(exercise_prediction) $ bandit -r -x tests,integration_tests src

Unit tests
~~~~~~~~~~

Quickstart
^^^^^^^^^^

To perform the code testing, simply run either:

.. code-block:: console

    (exercise_prediction) $ make code_tests

or

.. code-block:: console

    (exercise_prediction) $ coverage run -m pytest tests/
    (exercise_prediction) $ scoverage report -m

The details
^^^^^^^^^^^

The tests will likely not pass until the following criteria are met:

1. The dataset is converted from a MATLAB file to a series of PARQUET files via: ``make data`` (this creates the *datafile_group_splits.json* file necessary for validation).
2. The data is featurized via: ``make features`` (this provides the potentially system-specific ``FEATURIZE_ID`` for the featurization process).
3. ``FEATURIZE_ID`` is added as an environment variable in *.env*.

Integration tests
~~~~~~~~~~~~~~~~~

Quickstart
^^^^^^^^^^

To perform integration testing of the containerized streaming service using localstack,
simply run:

.. code-block:: console

    $ make integration_tests

The details
^^^^^^^^^^^

The ``make`` command runs *integration_tests/run.sh*, which performs the following steps:

1. Check for a local image of the streaming service container, and build one if it doesn't exist
2. Start the container and localstack to test the kinesis stream
3. Create a predictions output stream
4. Runs *integration_tests/test_docker.py* to test sending sample data to the service and checking the predicted response.
5. Runs *integration_tests/test_kinesis.py* to read from the output stream and check it against expected value.

If either of the tests fail, the logs are exported to the console.

Fin.
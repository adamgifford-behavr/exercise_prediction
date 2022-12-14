Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Syncing data to S3
^^^^^^^^^^^^^^^^^^

* `make test_environment` will test the basic python environment is set up correctly
* `make create_environment` will create an empty virtual environment with either ``conda``
  or ``virtualenv``
* `make requirements` will install all python dependencies for the project
* `make setup` to alternatively install dev requirements and create a virtual environment
  with ``pipenv``
* `create_stage_infra` to create staging infrastructure resourse with Terraform
* `create_prod_infra` to create production infrastructure resourse with Terraform
* `destroy_stage_infra` to destroy staging infrastructure resourse with Terraform
* `destroy_prod_infra` to destroy production infrastructure resourse with Terraform
* `make sync_data_to_s3` will use `aws s3 sync` to recursively sync files in `data/` up
  to your specified AWS S3 bucket (requires environment variables ``S3_BUCKET`` and
  ``AWS_CONFIG_PATH``, see :ref:`Getting Started: Installation <installation>`).
* `make sync_data_from_s3` will use `aws s3 sync` to recursively sync files from your
  specified AWS S3 bucket to `data/`.
* `make data` will prepare the dataset for featurization (requires that dataset is already
  downloaded [manual process] and in the correct location; see
  :ref:`Getting Started: Installation <installation>`)
* `make features` will build the features for model training
* `make stand_alone_train` or `make deploy_train` to run stand-alone model training or to
  deploy an orchestrated model training flow with Prefect, respectively
* `make stand_alone_score_batch` or `make deploy_score_batch` to run stand-alone batch model
  scoring or to deploy an orchestrated bactch scoring flow with Prefect, respectively
* `make deploy_web` to deploy the model as a web service
* `make deploy_streaming` to deploy the model as a streaming service
* `make docker_monitor` to run a simulated streaming monitoring service with Evidently
* `make quality_checks` to run code quality checks with ``isort``, ``black``, ``pylint``,
  ``mypy``, and ``bandit``
* `make code_tests` to run code testing with ``coverage`` and ``pytest``
* `make build` to build a Docker container image for the streaming service (will
  automatically call `make quality_checks` and `make code_tests`)
* `make integration_tests` to test the containerized streaming service with localstack
  (will automatically call `make build`)
* `make publish` to publish the containerized streaming service to Amazon ECR (will
  automatically call `make quality_checks`, `make code_tests`, `make build`, and
  `make integration_tests`)
* `make clean` to delete all compiled Python files

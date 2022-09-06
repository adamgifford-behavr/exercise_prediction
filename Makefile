.PHONY: clean data features requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# BUCKET = exercise-prediction/
# PROFILE = C:\Users\adamgifford_behavr.DESKTOP-PQU0D8M\.aws\config
BUCKET = ${S3_BUCKET}
PROFILE = ${AWS_CONFIG_PATH}
PROJECT_NAME = exercise_prediction

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt

## Make Dataset
data: requirements
	python src/data/make_dataset.py \
		data/raw/exercise_data.50.0000_multionly.mat \
		data/interim/ \
		src/data/val_split_crit.json \
		src/data/test_split_crit.json

## Build features
features: data
	python src/features/build_features.py \
		src/features/frequency_features.json \
		src/features/datafile_group_splits.json \
		src/features/metaparams.json


## run standalone training
stand_alone_train: features
	python src/models/train_model.py \
		naive_frequency_features \
		label_group \
		src/models/model_search.json

## orchestrate training
deploy_train: features
	cd src/orchestration
	prefect deployment build \
		orchestrate_train.py:train_flow \
		-n 'Main Model-Training Flow' \
		-q 'scheduled_training_flow'
	prefect deployment apply .\train_flow-deployment.yaml
	prefect agent start -q 'scheduled_training_flow'

## run standalone training
stand_alone_score_batch: stand_alone_train
	python src/models/score_batch.py

## orchestrate batch scoring
deploy_score_batch: deploy_train
	prefect deployment build \
		src/orchestration/orchestrate_score_batch.py:score_flow \
		-n 'Main Model-Scoring Flow' \
		-q 'scheduled_scoring_flow'
	prefect deployment apply .\score_flow-deployment.yaml
	prefect agent start -q 'scheduled_scoring_flow'

## simulate streaming monitoring service
docker_monitor:
	cd src/monitor
	python prepare.py
	docker-compose up

## code quality checks
quality_checks:
	isort src
	black src
	pylint src
	mypy src
	bandit -r src

## code testing
code_tests:
	coverage run -m pytest tests/ --disable-warnings
	coverage report -m

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
		conda create --name $(PROJECT_NAME) python=3.10
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	python -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=python3"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is set up correctly
test_environment:
	python test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')

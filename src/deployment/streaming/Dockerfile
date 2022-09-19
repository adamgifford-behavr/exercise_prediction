FROM public.ecr.aws/lambda/python:3.9

ARG AWS_DEFAULT_REGION="us-east-1"
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}

RUN pip install -U pip
RUN pip install pipenv

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY lambda_function.py .
COPY models ./models
COPY frequency_features.json .

CMD [ "lambda_function.lambda_handler" ]

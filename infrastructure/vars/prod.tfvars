source_stream_name                       = "prod_signals_stream"
output_stream_name                       = "prod_predictions_stream"
model_bucket                             = "prod-agifford-mlflow-artifacts-remote"
lambda_function_local_path               = "../src/deployment/streaming/lambda_function.py"
docker_image_local_path                  = "../src/deployment/streaming/Dockerfile"
ecr_repo_name                            = "prod_naive_feats_orch_cloud"
lambda_function_name                     = "prod_prediction_lambda"
lambda_event_source_mapping_batch_size   = 151
lambda_event_source_mapping_batch_window = 3

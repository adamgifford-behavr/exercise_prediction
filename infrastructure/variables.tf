variable "aws_region" {
  description = "AWS region to create resources"
  default     = "eu-east-1"
}

variable "project_id" {
  description = "project_id"
  default     = "exercise-prediction"
}

variable "source_stream_name" {
  description = ""
  default     = "signals_stream"
}

variable "output_stream_name" {
  description = ""
  default     = "predictions_stream"
}

variable "model_bucket" {
  description = "s3_bucket"
  default     = "agifford-mlflow-artifacts-remote"
}

variable "lambda_function_local_path" {
  description = ""
  default     = "src/deployment/streaming/"
}

variable "docker_image_local_path" {
  description = ""
}

variable "ecr_repo_name" {
  description = ""
  default     = "exercise_prediction_naive_feats_orch_cloud"
}

variable "lambda_function_name" {
  description = ""
  default     = "lambda_function.py"
}

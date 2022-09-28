variable "aws_region" {
  description = "AWS region to create resources"
  default     = "us-east-1"
}

variable "project_id" {
  description = "project_id"
  default     = "exercise-prediction"
}

variable "source_stream_name" {
  description = ""
}

variable "output_stream_name" {
  description = ""
}

variable "model_bucket" {
  description = "s3_bucket"
}

variable "lambda_function_local_path" {
  description = ""
}

variable "docker_image_local_path" {
  description = ""
}

variable "ecr_repo_name" {
  description = ""
}

variable "lambda_function_name" {
  description = ""
}

variable "lambda_event_source_mapping_batch_size" {
  description = "Lamdba maximum batch size"
}

variable "lambda_event_source_mapping_batch_window" {
  description = "Lambda maximum batch window to buffer samples"
}

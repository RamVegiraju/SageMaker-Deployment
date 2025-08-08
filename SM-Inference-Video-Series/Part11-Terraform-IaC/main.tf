provider "aws" {
  region = "us-east-1"
}

variable "sm-iam-role" {
    type = string
    default = "Replace this with your IAM Role"
    description = "The IAM Role for SageMaker Endpoint Deployment"
}

variable "container-image" {
    type = string
    default = "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.29.0-cpu-full"
    description = "The container you are utilizing for your SageMaker Model"
}

variable "model-data" {
    type = string
    default = "Replace this with your model data/artifacts"
    description = "The pre-trained model data/artifacts"
}

variable "instance-type" {
    type = string
    default = "ml.m5.xlarge"
    description = "The instance behind the SageMaker Real-Time Endpoint"
}



## Define your resources/building blocks here

# SageMaker Model Object
resource "aws_sagemaker_model" "sagemaker_model" {
  name = "sagemaker-model-sklearn"
  execution_role_arn = var.sm-iam-role

  primary_container {
    image = var.container-image
    mode = "SingleModel"
    model_data_url = var.model-data 
    environment = {
      "SAGEMAKER_PROGRAM" = "inference.py"
      "SAGEMAKER_SUBMIT_DIRECTORY" = var.model-data
    }
  }

  tags = {
    Name = "sagemaker-model-terraform"
  }
}


# Create SageMaker endpoint configuration
resource "aws_sagemaker_endpoint_configuration" "sagemaker_endpoint_configuration" {
  name = "sagemaker-endpoint-configuration-sklearn-tf"

  production_variants {
    initial_instance_count = 1
    instance_type = var.instance-type
    model_name = aws_sagemaker_model.sagemaker_model.name
    variant_name = "AllTraffic"
  }

  tags = {
    Name = "sagemaker-endpoint-configuration-terraform"
  }
}

# Create SageMaker Real-Time Endpoint
resource "aws_sagemaker_endpoint" "sagemaker_endpoint" {
  name = "sagemaker-endpoint-sklearn-tf"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.sagemaker_endpoint_configuration.name

  tags = {
    Name = "sagemaker-endpoint-terraform"
  }

}
import boto3
import json
import os
import tarfile
import subprocess
import sagemaker


#Setup
client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')
region = boto_session.region_name
print(region)
sagemaker_session = sagemaker.Session()

#Build tar file with model data + inference code
bashCommand = "tar -cvpzf model.tar.gz model.joblib model.py serving.properties requirements.txt"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

#Bucket for model artifacts
default_bucket = sagemaker_session.default_bucket()
print(default_bucket)

#Upload tar.gz to bucket
model_artifacts = f"s3://{default_bucket}/djl-sme-terraform/model.tar.gz"
response = s3.meta.client.upload_file('model.tar.gz', default_bucket, 'djl-sme-terraform/model.tar.gz')

print(f"Uploaded model data to: {model_artifacts}")
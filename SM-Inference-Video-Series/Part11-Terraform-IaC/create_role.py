import boto3
import json

# Initialize IAM client
iam = boto3.client('iam')

# Role name
role_name = "SageMakerExecutionRole"

# Assume role policy that lets SageMaker assume the role
assume_role_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}

# 1. Create the IAM Role
try:
    create_role_response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy),
        Description="Execution role for SageMaker to create and run models",
    )
    print(f"Role created: {create_role_response['Role']['Arn']}")
except iam.exceptions.EntityAlreadyExistsException:
    print(f"Role {role_name} already exists.")

# 2. Attach managed policies for SageMaker
#    This one gives full SageMaker access and S3 read/write (for model artifacts)
policies = [
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3FullAccess"
]

for policy_arn in policies:
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn=policy_arn
    )
    print(f"Attached policy: {policy_arn}")

# 3. Get the role ARN to use in create_model
role_arn = iam.get_role(RoleName=role_name)['Role']['Arn']
print("Execution Role ARN:", role_arn)
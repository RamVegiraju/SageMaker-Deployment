# [Video Coming]()
How to use CloudFormation and IaC best practices to deploy a SageMaker endpoint.

## AWS CLI command

```
# Replace with your stack name

aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name cfn-sm \
  --capabilities CAPABILITY_NAMED_IAM
```

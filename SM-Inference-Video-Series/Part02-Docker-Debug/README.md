# [YouTube Video](https://www.youtube.com/watch?v=UQHufr-DToE)

# Setup/Environment
This was run in c5.4xlarge SageMaker Classic Notebook Instance, Docker comes pre-installed here, optionally you can also use Studio and install Docker with the following guide which contains the required script: https://aws.amazon.com/blogs/machine-learning/accelerate-ml-workflows-with-amazon-sagemaker-studio-local-mode-and-docker-support/.

# Additional Notes/Considerations
Note that it's not best practice to have the model artifacts files (joblib, etc) exposed like this, it's best to run your own model training scripts and generate this file if you would like to mock your own model with this code/setup.

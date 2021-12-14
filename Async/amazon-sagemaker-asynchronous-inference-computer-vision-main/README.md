##  Run computer vision inference on large videos with Amazon SageMaker asynchronous endpoints

Associated blog: https://aws.amazon.com/blogs/machine-learning/run-computer-vision-inference-on-large-videos-with-amazon-sagemaker-asynchronous-endpoints/

In this sample, we serve a PyTorch Computer Vision model with SageMaker asynchronous inference endpoints to process a burst of traffic of large input payload videos. We demonstrate the new capabilities of an internal queue with user defined concurrency and completion notifications. We configure autoscaling of instances including  scaling down to 0 when traffic subsides and scale back up as the request queue fills up. We use SageMaker’s pre-built TorchServe container with a custom inference script for preprocessing the videos before model invocation. 
1. Large payload input of a high resolution video segment of 70 MB
2. Large payload output from a PyTorch pre-trained mask-rcnn model 
3. Large response time from the model of 30 seconds on a gpu instance
4. Auto-queuing of inference requests with asyncrhonous inference
5. Notifications of completed requests via SNS 
6. Auto-scaling of endpoints based on queue length metric with minimum value set to 0 instances

![Async_Diagram](https://user-images.githubusercontent.com/8871432/133468905-c0600795-b69a-4112-9777-3a4cd79c494c.png)


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.


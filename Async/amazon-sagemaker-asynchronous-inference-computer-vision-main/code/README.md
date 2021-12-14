SageMaker asynchronous inference - video processing with a pytorch mask-rcnn model  
input payload is a 70 mb video, 
output payload is bounding boxes, labels and scores for every sampled frame in the video, 
model response time is approx 40 seconds for 1 FPS sampling rate


1. Demonstrate a burst of incoming requests beyond the acceptable threshold and automatic queuing of the jobs and completion status notification (In the performance section of the blog, we identify 300 requests per minute as the threshold for a single endpoint )
2. Demonstrate scale down of resources when they are no incoming requests for ‘x’ minutes

         1. S3 input and output payload
        2. Autoqueueing
        3. Concurrency
        4. Autoscaling 
        5. Scale resources to 0 
        6. Notifications
        7. Cloud watch metrics
        8. Results - invocations, queue size, throughput





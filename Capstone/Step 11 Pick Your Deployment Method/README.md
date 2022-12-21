# Step 11: Pick Your Deployment Method

To deploy the model as Web API:

###1. Create an AWS EC2 instance

Follow this tutorial from AWS to create and launch an Amazon EC2 instance. A few customized settings for this project:

Step 1: Choose an Amazon Machine Image (AMI), choose the Deep Learning AMI (Ubuntu) AMI. Using this image does introduce a bit of extra overhead, however, it guarantees us that git and Docker will be pre-installed which saves a lot of trouble.
Step 2: Choose an Instance Type, choose t2.medium to ensure we have enough space to build and run our Docker image.
Step 3: Configure Security Group, choose Add Rule and create a custom tcp rule for port 8501 to make our streamlit app publicly available.
After clicking Launch, choose Create a new key pair, input "ec2-transformer", and click "Download Key Pair" to save ec2-transformer.pem key pair locally.

###2. Running Docker container in cloud
Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker’s methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production. Here I also deployed the trained transformer model using Docker on Amazon EC2 instance.

```bash
ssh -i ec2-gpt2-streamlit-app.pem ec2-user@your-instance-DNS-address.us-east-1.compute.amazonaws.com
```

Then, copy the code into the cloud using git:

git clone https://github.com/vveizhang/Bitcoin_Social_Media_Sentiment_Analysis.git
Afterwards, go into the ec2-docker folder to build and run the image:
```bash
cd ec2-docker/
docker image build -t streamlit:BertSentiment .
```





import os

AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
S3_BUCKET = os.getenv("S3_BUCKET", "deliver-bucket-24")
S3_PREFIX = os.getenv("S3_PREFIX", "client/")

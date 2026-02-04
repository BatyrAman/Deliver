import boto3
from botocore.exceptions import ClientError
from config import AWS_REGION, S3_BUCKET

class S3Uploader:
    def __init__(self):
        self.s3 = boto3.client("s3", region_name=AWS_REGION)

    def upload_bytes(self, data: bytes, key: str, content_type: str = "application/octet-stream"):
        try:
            self.s3.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return True
        except ClientError as e:
            print("[S3] upload error:", e)
            return False

    def upload_file(self, local_path: str, key: str, content_type: str = "application/octet-stream"):
        try:
            self.s3.upload_file(
                Filename=local_path,
                Bucket=S3_BUCKET,
                Key=key,
                ExtraArgs={"ContentType": content_type},
            )
            return True
        except ClientError as e:
            print("[S3] upload error:", e)
            return False

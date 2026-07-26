import json
import os
import boto3

sqs = boto3.client("sqs", region_name=os.environ.get("AWS_REGION", "us-east-1"))
TASKS_QUEUE_URL = os.environ["TASKS_QUEUE_URL"]


def lambda_handler(event, context):
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        print(f"New file detected: {key} in bucket {bucket}")

        sqs.send_message(
            QueueUrl=TASKS_QUEUE_URL,
            MessageBody=json.dumps({
                "bucket": bucket,
                "key": key
            })
        )

    return {"statusCode": 200, "body": "Message(s) sent to queue"}

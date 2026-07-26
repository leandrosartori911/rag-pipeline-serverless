import boto3
import json
import os
import requests
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "rag-pipeline")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TASKS_QUEUE_URL = os.environ["TASKS_QUEUE_URL"]
RESULTS_QUEUE_URL = os.environ["RESULTS_QUEUE_URL"]
DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "rag-document-chunks")

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

session = boto3.session.Session(profile_name=AWS_PROFILE)
sqs = session.client("sqs", region_name=AWS_REGION)
s3 = session.client("s3", region_name=AWS_REGION)
table = session.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMODB_TABLE_NAME)

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


def get_embedding(text):
    response = requests.post(OLLAMA_EMBED_URL, json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return response.json()["embedding"]


def handle_ingest(body):
    bucket, key = body["bucket"], body["key"]
    print(f"[INGEST] Processing: {key}")

    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        table.put_item(Item={
            "document_id": key,
            "chunk_index": i,
            "chunk_text": chunk,
            "embedding": [Decimal(str(x)) for x in embedding]
        })
        print(f"  Chunk {i} saved")
    print("[INGEST] Document processed.\n")


def handle_query(body):
    query_id = body["query_id"]
    query_text = body["text"]
    print(f"[QUERY] Generating embedding for: {query_text}")

    embedding = get_embedding(query_text)

    sqs.send_message(
        QueueUrl=RESULTS_QUEUE_URL,
        MessageBody=json.dumps({
            "query_id": query_id,
            "embedding": embedding
        })
    )
    print(f"[QUERY] Embedding returned for query_id={query_id}\n")


def handle_generate(body):
    query_id = body["query_id"]
    prompt = body["prompt"]
    print(f"[GENERATE] Generating answer for query_id={query_id}")

    response = requests.post(OLLAMA_GENERATE_URL, json={
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False
    })
    answer = response.json().get("response", "")

    sqs.send_message(
        QueueUrl=RESULTS_QUEUE_URL,
        MessageBody=json.dumps({
            "query_id": query_id,
            "answer": answer
        })
    )
    print(f"[GENERATE] Answer returned for query_id={query_id}\n")


def poll():
    print("Worker running. Ctrl+C to stop.")
    while True:
        response = sqs.receive_message(
            QueueUrl=TASKS_QUEUE_URL, MaxNumberOfMessages=1, WaitTimeSeconds=20
        )
        messages = response.get("Messages", [])
        if not messages:
            continue

        msg = messages[0]
        body = json.loads(msg["Body"])

        try:
            if body.get("type") == "query":
                handle_query(body)
            elif body.get("type") == "generate":
                handle_generate(body)
            else:
                handle_ingest(body)

            sqs.delete_message(QueueUrl=TASKS_QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])

        except Exception as e:
            print(f"[ERROR] Failed to process message: {e}")
            print("Message NOT removed from queue — will reappear after VisibilityTimeout.\n")


if __name__ == "__main__":
    poll()

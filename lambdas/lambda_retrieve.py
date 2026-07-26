import json
import os
import boto3
import uuid
import time

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
sqs = boto3.client("sqs", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(os.environ.get("DYNAMODB_TABLE_NAME", "rag-document-chunks"))

TASKS_QUEUE_URL = os.environ["TASKS_QUEUE_URL"]
RESULTS_QUEUE_URL = os.environ["RESULTS_QUEUE_URL"]

TOP_N = 3
MAX_WAIT_SECONDS = 25


def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a ** 2 for a in vec_a) ** 0.5
    magnitude_b = sum(b ** 2 for b in vec_b) ** 0.5
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)


def wait_for_result(expected_query_id, deadline):
    while time.time() < deadline:
        response = sqs.receive_message(
            QueueUrl=RESULTS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=5
        )
        messages = response.get("Messages", [])
        if not messages:
            continue

        msg = messages[0]
        result_body = json.loads(msg["Body"])

        if result_body.get("query_id") == expected_query_id:
            sqs.delete_message(QueueUrl=RESULTS_QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
            return result_body
    return None


def lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))
    question = body.get("question", "")

    if not question:
        return {"statusCode": 400, "body": json.dumps({"error": "'question' field is required"})}

    # 1. Request the question's embedding
    embed_query_id = str(uuid.uuid4())
    sqs.send_message(
        QueueUrl=TASKS_QUEUE_URL,
        MessageBody=json.dumps({
            "type": "query",
            "query_id": embed_query_id,
            "text": question
        })
    )

    embed_result = wait_for_result(embed_query_id, time.time() + MAX_WAIT_SECONDS)
    if embed_result is None:
        return {"statusCode": 504, "body": json.dumps({"error": "timeout waiting for embedding"})}

    question_embedding = embed_result["embedding"]

    # 2. Scan DynamoDB and compute similarity
    scan_response = table.scan()
    chunks = scan_response["Items"]

    scored_chunks = []
    for chunk in chunks:
        chunk_embedding = [float(x) for x in chunk["embedding"]]
        similarity = cosine_similarity(question_embedding, chunk_embedding)
        scored_chunks.append({
            "document_id": chunk["document_id"],
            "chunk_index": int(chunk["chunk_index"]),
            "chunk_text": chunk["chunk_text"],
            "similarity": similarity
        })

    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    top_chunks = scored_chunks[:TOP_N]

    # 3. Build the prompt with retrieved context
    context = "\n\n".join([c["chunk_text"] for c in top_chunks])
    prompt = f"""Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    # 4. Request answer generation
    generate_query_id = str(uuid.uuid4())
    sqs.send_message(
        QueueUrl=TASKS_QUEUE_URL,
        MessageBody=json.dumps({
            "type": "generate",
            "query_id": generate_query_id,
            "prompt": prompt
        })
    )

    generate_result = wait_for_result(generate_query_id, time.time() + MAX_WAIT_SECONDS)
    if generate_result is None:
        return {"statusCode": 504, "body": json.dumps({"error": "timeout waiting for generation"})}

    return {
        "statusCode": 200,
        "body": json.dumps({
            "question": question,
            "answer": generate_result["answer"],
            "sources": top_chunks
        })
    }

# Hybrid Serverless RAG Pipeline (AWS + Local LLMs)

A Retrieval-Augmented Generation (RAG) system built entirely on AWS serverless primitives, with all embedding and text generation running **locally** on consumer GPU hardware — connected to the cloud through an asynchronous, zero-inbound-exposure architecture.

**Zero cost. Zero open ports. Zero cloud LLM billing.**

---

## Why this project exists

Most RAG tutorials assume you'll pay for a managed vector database (OpenSearch, Pinecone) and a hosted LLM API (Bedrock, OpenAI). This project deliberately avoids both, to explore what a RAG pipeline looks like when:

1. **Cost must be $0, permanently** — not "free for 12 months," not "free tier with a credit card on file."
2. **Compute-heavy inference runs on hardware you already own** (a home GPU) instead of a metered cloud service.
3. **That home hardware is never exposed to the public internet** — no open inbound ports, no reverse proxy, no attack surface added.

Every architectural decision below is a direct consequence of one of these three constraints, and each was validated against official AWS pricing documentation before being adopted — not assumed.

---

## Architecture

```
┌─────────────┐     upload      ┌──────────┐   S3 event   ┌──────────────────┐
│   Document   │ ──────────────▶ │    S3    │ ───────────▶ │  Lambda (ingest)  │
└─────────────┘                  └──────────┘              └────────┬──────────┘
                                                                     │ sends task
                                                                     ▼
                                                            ┌──────────────────┐
                                                            │  SQS (tasks queue) │
                                                            └────────┬──────────┘
                                                                     │ long-polling
                                                                     ▼
                                                       ┌───────────────────────────┐
                                                       │   Local Worker (Python)    │
                                                       │   running on home GPU      │
                                                       │   (RTX 5060)               │
                                                       │                            │
                                                       │  → Ollama: nomic-embed-text│
                                                       │  → Ollama: llama3.1:8b     │
                                                       └─────────────┬──────────────┘
                                                                     │ writes vectors
                                                                     ▼
                                                            ┌──────────────────┐
                                                            │    DynamoDB       │
                                                            │ (chunks + vectors)│
                                                            └────────┬──────────┘
                                                                     ▲
                                                                     │ manual cosine
                                                                     │ similarity scan
┌─────────────┐   question      ┌──────────────────┐   query task   │
│     User     │ ──────────────▶ │ Lambda (retrieve) │ ──────────────┘
└─────────────┘                  │  Function URL     │
       ▲                         └────────┬──────────┘
       │ answer + sources                  │ (round-trips through
       └──────────────────────────────────┘  SQS results queue to
                                             reach the local worker
                                             for embedding + generation)
```

**Ingestion flow:** Upload to S3 → S3 event triggers Lambda → Lambda enqueues a task in SQS → local worker polls, downloads the document, chunks it, generates embeddings via Ollama, writes chunk + vector to DynamoDB.

**Query flow:** User sends a question to the Lambda Function URL → Lambda enqueues an embedding request → local worker generates the question's embedding and returns it via a results queue → Lambda scans DynamoDB and computes cosine similarity in-process → Lambda builds a prompt from the top matching chunks → enqueues a generation request → local worker calls the local LLM → Lambda returns the final answer with source chunks attached.

---

## Key architectural decisions

| Decision | What was avoided | Why |
|---|---|---|
| **SQS task/result queues instead of direct inbound calls to the home PC** | Tailscale Funnel / any public-facing tunnel into the local machine | The local machine only ever makes *outbound* polling calls. No inbound port is ever opened — the attack surface of the home PC is unchanged from before the project existed. This was a hard requirement, not a preference: an earlier design using Tailscale Funnel was rejected specifically because it exposed a port publicly, even with an API key in front of it. |
| **DynamoDB with manual cosine similarity in Lambda, instead of OpenSearch** | Amazon OpenSearch (managed vector store) | OpenSearch has a minimum hourly charge even when idle — it does not fit an "always free" model. DynamoDB's on-demand pricing and Always Free tier (25 GB storage, 25 RCU/WCU) has no such floor. The trade-off is explicit and accepted: similarity search is a full table scan computed in Lambda, not an indexed vector search — fine at portfolio scale (dozens–hundreds of chunks), and documented as a known limitation rather than hidden. |
| **Local LLM inference (Ollama) instead of Amazon Bedrock** | Bedrock (pay-per-token hosted models) | Bedrock has no meaningful permanent free tier — every token is billed. Running `llama3.1:8b` and `nomic-embed-text` locally on an RTX 5060 makes inference cost exactly $0, at the expense of throughput and needing the local machine powered on. |
| **Lambda Function URL instead of API Gateway** | API Gateway's 12-month free tier | API Gateway's free allotment (1M REST calls) is part of the *12-month* free tier group, which expires silently and starts billing. Lambda Function URLs are a built-in, no-additional-cost way to expose a Lambda over HTTP. |
| **Dedicated, least-privilege IAM identities per component** | A single admin/root credential reused everywhere | A separate IAM user (CLI operations) and separate IAM roles (one per Lambda function) were created, each with a hand-written policy scoped to only the specific resources (`rag-*` ARNs) and actions that component actually needs. A leaked credential from any single component cannot touch the rest of the account. |
| **Asynchronous, queue-based communication instead of synchronous request/response** | A traditional "API calls GPU directly and waits" design | This is the direct cost of the zero-exposure requirement: since the home GPU can't accept inbound connections, every request becomes a round-trip through SQS. Latency increases (~5–8 seconds end-to-end for a full query+generation cycle) in exchange for zero network exposure — a deliberate, documented trade-off, not an oversight. |

---

## AWS Free Tier research (a real part of this project)

Cost verification wasn't an assumption — it was checked against AWS's own documentation before building on each service, because "free tier" claims found in third-party blog posts turned out to be inconsistent or outdated:

- AWS's own FAQ states that **Amazon S3, EC2, and CloudFront are 12-month free**, not Always Free — this directly contradicted several third-party sources claiming permanent S3 free tier for new accounts. S3 usage in this project is treated as consuming trial credit, not as permanently free.
- Accounts created after **July 15, 2025** (this account included) fall under AWS's newer Free Plan model: a capped credit balance with a hard 6-month/credit-exhaustion expiry window, rather than the older blanket 12-month trial. This was confirmed directly from the account's Billing dashboard.
- **Lambda, DynamoDB, and SQS were individually confirmed as "Always Free"** — permanent, uncapped-by-time free tiers that apply regardless of account age (1M Lambda requests/month, 25 GB DynamoDB storage, 1M SQS requests/month).

This distinction — *time-limited trial credit* vs. *permanent Always Free* — directly shaped which services were selected for anything running long-term versus anything only used transiently (like S3, used only for ingestion staging).

---

## Evaluation: measuring quality with RAGAS (not just eyeballing it)

RAG systems can look correct while quietly hallucinating. To catch this, the pipeline was evaluated with [RAGAS](https://github.com/explodinggradients/ragas), configured to use **the same local LLM as the judge** — via Ollama's OpenAI-compatible endpoint — so evaluation itself also runs at zero cost.

Two real test cases were evaluated:

| Test case | Faithfulness | Context Precision | Context Recall |
|---|---|---|---|
| Real content (submarine cable infrastructure document) | 0.50 | 1.00 | 1.00 |
| Placeholder content (Lorem Ipsum, no real meaning) | **0.00** | 0.00 | N/A |

The second row is the key result: the system had already been observed, by manual inspection, to hallucinate an answer ("about people's rights and data protection") when asked a question against meaningless placeholder text — the LLM ignored its own "answer only from the provided context" instruction. RAGAS's faithfulness score of **0.0** confirms this numerically rather than anecdotally, demonstrating that the evaluation harness correctly detects unfaithful generation. This is treated as a feature of the evaluation setup being validated, not a flaw to hide — a portfolio project that only shows passing metrics is less convincing than one that shows the measurement tooling catching a real failure.

The first row also surfaced a real limitation of the naive retrieval approach: with only a handful of chunks in the table, a semantically adjacent-but-not-most-relevant chunk occasionally scored a higher cosine similarity than the chunk that actually answered the question — the LLM still answered correctly by reading past the noise, but it highlights why `context_precision` matters as a separate signal from whether the final answer happens to be right.

---

## Known limitations (by design, not oversight)

- **Retrieval is a full table scan**, not an indexed ANN search. This is the direct trade-off of avoiding OpenSearch. It works fine at the scale tested; it would not scale to a production corpus of millions of chunks without revisiting this decision.
- **The system is only available while the local machine is powered on** and Ollama is running — this is inherent to the "compute happens at home" design, not a bug.
- **End-to-end latency is ~5–8 seconds** per query (two SQS round-trips: one for the question embedding, one for generation), noticeably slower than a synchronous cloud API. This is the accepted cost of never exposing the local machine to inbound traffic.
- **S3 usage consumes trial credit**, not permanent free tier, on this AWS account. The project is designed to be fully built and demonstrated within the credit window; keeping it running indefinitely past the account's free-plan expiry would require moving to a paid plan.
- **Cold-start latency on first LLM call**: Ollama unloads an idle model from VRAM after a timeout, so the first request after a period of inactivity takes noticeably longer (~1 minute) than subsequent ones.

---

## Tech stack

- **AWS**: S3, Lambda, SQS, DynamoDB, IAM — all serverless, all Always-Free-tier-eligible except S3
- **Local inference**: [Ollama](https://ollama.com/), running `nomic-embed-text` (embeddings) and `llama3.1:8b` (generation) on an RTX 5060
- **Evaluation**: [RAGAS](https://github.com/explodinggradients/ragas), configured for a local, OpenAI-API-compatible judge
- **Language**: Python 3 (worker + evaluation scripts), Python 3.13 (Lambda runtime)
- **Libraries**: `boto3`, `requests`, `ragas`, `openai` (client only, pointed at a local endpoint)

---

## How it works, end to end

1. A document is uploaded to an S3 bucket.
2. An S3 event notification triggers a Lambda function, which enqueues an ingestion task in SQS — no processing happens in the Lambda itself.
3. A Python worker running on local hardware long-polls the SQS queue, downloads the document from S3, splits it into overlapping ~400-word chunks, and calls the local embedding model for each chunk.
4. Each chunk and its 768-dimension embedding vector is written to DynamoDB, keyed by `document_id` (partition key) and `chunk_index` (sort key).
5. A user submits a question to a Lambda Function URL.
6. The Lambda enqueues an embedding request for the question and waits (with a timeout) for the local worker to compute and return it via a results queue.
7. The Lambda scans the DynamoDB table, computes cosine similarity between the question's vector and every stored chunk vector, and selects the top matches.
8. The Lambda constructs a prompt instructing the LLM to answer strictly from the retrieved context, enqueues a generation request, and waits for the local worker to run it through the local LLM.
9. The final answer, along with the source chunks used, is returned to the user.

---

## Setup

Requires: an AWS account, [Ollama](https://ollama.com/) installed locally, [Tailscale](https://tailscale.com/) (used during early architecture exploration; not required by the final design), and Python 3.

```bash
# Pull the required local models
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# Install worker dependencies
pip install boto3 requests

# Configure an isolated AWS CLI profile (least-privilege IAM user)
aws configure --profile rag-pipeline

# Run the local worker
python worker.py
```

AWS-side resources (S3 bucket, SQS queues, DynamoDB table, Lambda functions, IAM roles/policies) are provisioned as described in the architecture section above.

---

## What this project demonstrates

- Designing under a **hard, non-negotiable cost constraint**, including researching and citing official pricing sources rather than trusting secondhand claims
- **Least-privilege IAM** applied consistently across every component, not just at the account level
- A working **hybrid cloud/edge architecture** where expensive compute runs on owned hardware, orchestrated by serverless cloud primitives
- Building an **asynchronous system by necessity** (no inbound exposure allowed) and correctly reasoning through the latency trade-offs that decision creates
- **Quantitative RAG evaluation**, not just manual spot-checking — including a documented case where the evaluation framework caught a real hallucination
- Debugging real-world Python dependency conflicts in an isolated `venv`, without letting the fix contaminate an unrelated existing project's environment

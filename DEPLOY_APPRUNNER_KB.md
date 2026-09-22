# Deploying with App Runner + Bedrock Knowledge Bases

This replaces the local numpy pipeline (`api.py`, `rag/ingest.py`,
`rag/store.py`) with a fully managed AWS pipeline: S3 for storage, a
Bedrock Knowledge Base for chunking/embedding/retrieval (Titan embeddings,
OpenSearch Serverless under the hood), and App Runner instead of a server
you manage. Generation stays on Gemini via `rag/generate.py`, unchanged.

Your original `api.py` still works exactly as before — this guide uses the
new `api_kb.py` and `Dockerfile.apprunner` instead, so nothing here breaks
what you already have running on EC2.

**Order matters here more than in the other guides** — Knowledge Bases
depend on the S3 bucket and IAM role existing first, and App Runner needs
the Knowledge Base ID before it can start. Follow the steps in sequence.

---

## Step 1 — Create the S3 bucket

1. [S3 Console](https://console.aws.amazon.com/s3/) → **Create bucket**
2. Name it something globally unique, e.g. `rag-project-docs-yourname`
3. Leave everything else default → **Create bucket**

This holds your raw PDFs/text files — the Knowledge Base reads from here.

## Step 2 — Create the Knowledge Base

This is the biggest step, but AWS's wizard does most of the hard
infrastructure work (OpenSearch Serverless collection, vector index, IAM
service role) for you automatically.

1. [Bedrock Console](https://console.aws.amazon.com/bedrock/) →
   **Knowledge Bases** (left sidebar, under Build) → **Create**
2. **Name**: `rag-project-kb`
3. **IAM permissions**: choose **Create and use a new service role** — this
   is a *different* role from the one your app uses later; this one lets
   the Knowledge Base itself read S3 and write to OpenSearch
4. **Data source**: Amazon S3 → select the bucket from Step 1
5. **Embeddings model**: **Titan Text Embeddings V2**
6. **Vector database**: choose **Quick create a new vector store** →
   Amazon OpenSearch Serverless — this is what saves you from manually
   configuring OpenSearch security policies and a vector index by hand
7. Review and **Create Knowledge Base**. This takes a few minutes — AWS is
   provisioning the OpenSearch Serverless collection behind the scenes.
8. Once created, **copy the Knowledge Base ID** (looks like `ABCD1234EF`)
   and the **Data Source ID** shown on the same page — you'll need both
   shortly.

## Step 3 — Do an initial sync

1. Still on the Knowledge Base's page, under **Data source**, select it
   and click **Sync**
2. It'll say "Nothing to sync" if the bucket is empty — that's fine, this
   confirms the connection works. Once you upload documents through the
   app later, syncing happens automatically (the app triggers it via
   `start_ingestion_job()` in `rag/kb_client.py`).

## Step 4 — Store your Gemini key in Parameter Store

1. [Systems Manager Console](https://console.aws.amazon.com/systems-manager/parameters) → **Create parameter**
2. Name: `/rag-project/gemini-api-key`
3. Type: **SecureString**
4. Value: your `AIzaSy...` key
5. **Create parameter**

## Step 5 — Create the App Runner instance role

This is the role your *app* uses at runtime (separate from the Knowledge
Base's own service role from Step 2) — it needs permission to call the
Knowledge Base, read/write S3, and read the Gemini key from Parameter Store.

1. **IAM Console** → **Roles** → **Create role**
2. Trusted entity: **AWS service** → search for and select **App Runner**
   → choose the **Tasks** use case (this is the role your running
   container assumes, distinct from App Runner's build-time role)
3. Skip attaching a managed policy for now — click through to create the
   role, name it `rag-project-apprunner-role`
4. Open the role → **Add permissions** → **Create inline policy** → JSON tab, paste:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:Retrieve", "bedrock-agent:StartIngestionJob"],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::rag-project-docs-yourname",
        "arn:aws:s3:::rag-project-docs-yourname/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["ssm:GetParameter"],
      "Resource": "arn:aws:ssm:*:*:parameter/rag-project/*"
    }
  ]
}
```
(Replace `rag-project-docs-yourname` with your actual bucket name from Step 1.)

## Step 6 — Push the Docker image to ECR

This is the one step in this whole guide that isn't pure console clicking
— pushing a Docker image requires the Docker and AWS CLIs. If you don't
have them locally, **AWS CloudShell** (a terminal built into the console,
top-right icon that looks like `>_`) has both preinstalled — you can
upload your project zip there and run everything below without installing
anything on your own machine.

1. **ECR Console** → **Repositories** → **Create repository**
   - Name: `rag-project-kb`
   - Leave the rest default → **Create**
2. Click into the repository → **View push commands** — it shows the
   exact commands for your account, but they're roughly:
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -f Dockerfile.apprunner -t rag-project-kb .

docker tag rag-project-kb:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-project-kb:latest

docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-project-kb:latest
```
Note the `-f Dockerfile.apprunner` — this project has two Dockerfiles;
make sure you're building the right one.

## Step 7 — Create the App Runner service

1. [App Runner Console](https://console.aws.amazon.com/apprunner/) → **Create service**
2. **Source**: Container registry → **Amazon ECR** → browse to the
   `rag-project-kb` image you just pushed
3. **Deployment trigger**: Automatic (so future pushes to this ECR repo
   auto-redeploy) or Manual, your preference
4. **Port**: `8000`
5. **Environment variables** — this is where the pieces from Steps 1–4 connect:
   | Key | Value |
   |---|---|
   | `AWS_REGION` | `us-east-1` (or whichever region you used) |
   | `KNOWLEDGE_BASE_ID` | from Step 2 |
   | `DATA_SOURCE_ID` | from Step 2 |
   | `S3_BUCKET_NAME` | your bucket name from Step 1 |
   | `GEMINI_API_KEY_PARAM` | `/rag-project/gemini-api-key` |
6. **Instance role**: select `rag-project-apprunner-role` from Step 5 —
   this is what lets `boto3` inside the container call Bedrock/S3/SSM with
   zero credentials anywhere in your code or env vars
7. **Create & deploy**

App Runner gives you an HTTPS URL automatically (something like
`https://xxxxx.us-east-1.awsapprunner.com`) — no Elastic IP, nginx, or
certbot needed, unlike the EC2 guides.

## Step 8 — Test it

Open the App Runner URL. Upload a PDF or paste a blog URL through the same
UI you already know — this time it's landing in S3 and getting indexed by
the Knowledge Base instead of your local numpy array. Give it 30–60
seconds after upload before asking a question, since the sync isn't
instant even when triggered manually.

## Updating the app later

```bash
docker build -f Dockerfile.apprunner -t rag-project-kb .
docker tag rag-project-kb:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-project-kb:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-project-kb:latest
```
If you chose automatic deployment in Step 7, App Runner picks this up on
its own within a minute or two.

---

## Cost note

This is the most expensive of the three deployment options by a good
margin. OpenSearch Serverless bills for a minimum capacity (OCUs) even at
near-zero usage — check current pricing before leaving this running
long-term. App Runner itself is pay-per-use and cheap at low traffic; the
OpenSearch minimum is the number to watch.

## Troubleshooting

- **`/query` returns "Nothing relevant found" even after uploading** →
  check the Knowledge Base's Data Source sync history in the Bedrock
  Console — confirm the sync actually completed (not just started) before
  assuming retrieval is broken.
- **`AccessDeniedException` on `bedrock:Retrieve`** → the App Runner
  instance role (Step 5) is missing or its inline policy doesn't match
  your actual `KNOWLEDGE_BASE_ID`/bucket name.
- **App Runner build succeeds but the app 500s immediately** → almost
  always a missing/mistyped environment variable from Step 7's table —
  check App Runner's application logs (Logs tab on the service page).
- **`ParameterNotFound` fetching the Gemini key** → confirm
  `GEMINI_API_KEY_PARAM`'s *value* is the parameter's exact path
  (`/rag-project/gemini-api-key`), not the key itself — that env var holds
  a pointer, not the secret.

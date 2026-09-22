# Deploying with ECS Express Mode + Bedrock Knowledge Bases

This is a corrected version of `DEPLOY_APPRUNNER_KB.md`. **AWS App Runner
stopped accepting new customers on April 30, 2026** and was never
available in `ap-south-2` (Hyderabad) even before that — so it's not a
usable target for a new deployment right now.

Amazon **ECS Express Mode** is AWS's own recommended replacement: same
pitch as App Runner (give it a container image and a couple of IAM roles,
get a running HTTPS service with load balancing and auto-scaling, no
manual cluster/task-definition wrangling), but actively developed and open
to new customers.

Everything else from the original plan is unchanged — S3, the Bedrock
Knowledge Base, Titan embeddings, OpenSearch Serverless, and Parameter
Store all work exactly as before. Only the "where does the container
actually run" step is different. Steps 1–4 below are identical to the
App Runner guide; skip ahead if you already did them.

**About `ap-south-2` (Hyderabad) specifically**, since that's what you're
using: Bedrock itself and Titan embeddings both work there directly.
Claude Haiku 4.5, however, is only reachable from `ap-south-2` via
**Global Cross-Region Inference** — meaning `BEDROCK_MODEL_ID` needs a
CRIS-prefixed inference profile ID (something like
`global.anthropic.claude-haiku-4-5-20251001-v1:0`), not the bare
`anthropic.claude-haiku-4-5-20251001-v1:0` ID. Check the exact string
under **Bedrock Console → Infer → Cross-region inference** before setting
it in Step 7 — don't assume the bare ID will work, it likely won't.

Separately, whether **Knowledge Base creation with the OpenSearch
Serverless "quick create" option is offered in `ap-south-2` at all** isn't
something documented clearly enough to confirm here — you'll find out for
certain at Step 2. If that option doesn't appear, `ap-south-1` (Mumbai)
is the fallback with the longer track record; just keep every region
setting in this guide consistent with whichever you end up using.

---

## Step 1 — Create the S3 bucket

1. [S3 Console](https://console.aws.amazon.com/s3/) → **Create bucket**
2. Name it something globally unique, e.g. `rag-project-docs-yourname`
3. Leave everything else default → **Create bucket**

## Step 2 — Create the Knowledge Base

1. [Bedrock Console](https://console.aws.amazon.com/bedrock/) →
   **Knowledge Bases** → **Create**
2. **Name**: `rag-project-kb`
3. **IAM permissions**: **Create and use a new service role**
4. **Data source**: Amazon S3 → select the bucket from Step 1
5. **Embeddings model**: **Titan Text Embeddings V2**
6. **Vector database**: **Quick create a new vector store** → Amazon
   OpenSearch Serverless
7. **Create Knowledge Base** (takes a few minutes)
8. Copy the **Knowledge Base ID** and **Data Source ID** shown on the page

## Step 3 — Initial sync

Data source → **Sync**. "Nothing to sync" is expected on an empty bucket —
this just confirms the connection. Later uploads trigger sync
automatically via the app.

## Step 4 — Store your Gemini key in Parameter Store

1. [Systems Manager Console](https://console.aws.amazon.com/systems-manager/parameters) → **Create parameter**
2. Name: `/rag-project/gemini-api-key`, Type: **SecureString**, Value: your `AIzaSy...` key
3. **Create parameter**

## Step 5 — Create the task role (your app's AWS permissions)

This is the IAM role your *running container* uses — equivalent to what
would've been called the "instance role" under App Runner. Under ECS this
is called a **task role**.

1. **IAM Console** → **Roles** → **Create role**
2. Trusted entity: **AWS service** → search **Elastic Container Service**
   → choose **Elastic Container Service Task**
3. Skip attaching a managed policy → create the role, name it
   `rag-project-task-role`
4. Open the role → **Add permissions** → **Create inline policy** → JSON tab:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:Retrieve",
        "bedrock-agent:StartIngestionJob",
        "bedrock:InvokeModel",
        "bedrock:Converse"
      ],
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
(Replace `rag-project-docs-yourname` with your actual bucket name. The
`bedrock:InvokeModel`/`bedrock:Converse` actions are only needed if
`GENERATION_BACKEND=bedrock` — omit them if you're generating with Gemini
instead and only using Bedrock for retrieval.)

You'll pick this role as the **Task role** in Step 7 — don't confuse it
with the *Task execution role* ECS also asks for, which is a different,
AWS-managed role that just pulls your image and writes logs (Express Mode
can auto-create that one for you, no custom policy needed).

## Step 6 — Push the Docker image to ECR

Same as before — this is the one step that needs Docker/AWS CLI rather
than pure console clicking. Use **AWS CloudShell** (top-right `>_` icon in
the console) if you don't want to install anything locally.

1. **ECR Console** → **Repositories** → **Create repository** → name it
   `rag-project-kb` → **Create**
2. Click into it → **View push commands**, or run:
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -f Dockerfile.apprunner -t rag-project-kb .

docker tag rag-project-kb:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-project-kb:latest

docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-project-kb:latest
```
(The `Dockerfile.apprunner` name is a leftover from planning around App
Runner — the file itself is generic, just points `uvicorn` at `api_kb:app`.
No need to rename it.)

## Step 7 — Create the Express Mode service

1. [ECS Console](https://console.aws.amazon.com/ecs/) — the first-run
   experience should offer **Express Mode** directly; otherwise look for
   **Express Mode services** in the left sidebar → **Create**
2. **Service name**: `rag-project-kb`
3. **Container image**: browse to the `rag-project-kb` repository in ECR,
   select the `latest` tag (AWS recommends selecting by image digest for
   production, but the tag is fine while you're testing)
4. **Container port**: `8000`
5. **Health check path**: `/`
6. **Environment variables** — add each of these as a plain key/value
   (none are secrets themselves — `GEMINI_API_KEY_PARAM` is just a
   *pointer* to the real secret in Parameter Store):

   | Key | Value |
   |---|---|
   | `AWS_REGION` | your region, e.g. `ap-south-2` |
   | `KNOWLEDGE_BASE_ID` | from Step 2 |
   | `DATA_SOURCE_ID` | from Step 2 |
   | `S3_BUCKET_NAME` | your bucket name from Step 1 |
   | `GENERATION_BACKEND` | `bedrock` (or `gemini` — see region note above) |
   | `BEDROCK_MODEL_ID` | only if `GENERATION_BACKEND=bedrock` — verify the exact ID in the Bedrock Console first (Infer → Cross-region inference); some models need a CRIS-prefixed ID like `global.anthropic...` rather than the bare model ID |
   | `GEMINI_API_KEY_PARAM` | only if `GENERATION_BACKEND=gemini` — `/rag-project/gemini-api-key` |

7. **Task role**: select `rag-project-task-role` from Step 5 — this is
   what lets `boto3` inside the container call Bedrock/S3/SSM with zero
   credentials in your code
8. **Task execution role**: choose **Create new role** — Express Mode
   auto-creates this with the standard `AmazonECSTaskExecutionRolePolicy`,
   nothing custom needed
9. **Infrastructure role**: choose **Create new role** — this is what lets
   ECS itself provision the load balancer, security groups, and
   networking on your behalf (`AmazonECSInfrastructureRoleforExpressGatewayServices`)
10. **Create**

If you hit `Invalid Parameter Exception: Unable to assume the service
linked role` on your very first Express Mode service in this account,
that's a known first-run race condition — wait a few seconds and retry.

## Step 8 — Test it

Once deployed, ECS shows an **Application URL** on the service page
(HTTPS, backed by the auto-provisioned load balancer). Open it — same UI
as before. Upload a file or paste a URL, wait 30–60 seconds for the
Knowledge Base sync, then ask a question.

## Updating the app later

```bash
docker build -f Dockerfile.apprunner -t rag-project-kb .
docker tag rag-project-kb:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-project-kb:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/rag-project-kb:latest
```
Then in the ECS console, open your Express Mode service and use **Deploy**
/ **Force new deployment** to pick up the new image — pushing to ECR alone
doesn't auto-redeploy the way App Runner's automatic trigger did.

---

## What Express Mode is actually provisioning for you

Worth knowing since it's all visible (and editable) in the console
afterward: an ECS cluster running your container on **Fargate**, an
**Application Load Balancer** with an SSL/TLS listener (Express Mode can
consolidate multiple Express Mode services behind one shared ALB), security
groups scoped so only the ALB can reach your container, a CloudWatch log
group, and a target-tracking auto-scaling policy. That's the same
infrastructure list App Runner used to hide behind one button — Express
Mode just makes each piece visible and separately adjustable later.

## Troubleshooting

- **Same Bedrock/S3/Parameter Store issues as the App Runner guide** —
  `AccessDeniedException` on `bedrock:Retrieve` almost always means the
  task role (Step 5) is missing or its policy doesn't match your real
  `KNOWLEDGE_BASE_ID`/bucket name; `ParameterNotFound` means
  `GEMINI_API_KEY_PARAM`'s value isn't the exact parameter path.
- **Confusing the three IAM roles** — Express Mode asks for three
  different roles and it's easy to mix them up: *Task role* = your app's
  own AWS permissions (Step 5, the one you built by hand). *Task execution
  role* = lets ECS pull the image and write logs (auto-created, generic).
  *Infrastructure role* = lets ECS provision the ALB/networking around
  your service (auto-created, generic). Only the first one needs your
  custom policy.
- **Service stuck / first deployment fails immediately** → check the
  **Logs** tab on the service page for a Python traceback — usually a
  missing environment variable from Step 7's table.

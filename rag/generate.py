"""
generate.py — the generation step, swappable between Google Gemini and
Amazon Bedrock via the GENERATION_BACKEND env var (see .env.example).

Both api.py and query.py call generate_answer(prompt) and get plain text
back — neither needs to know or care which provider actually ran it. That's
the same "swap the LLM without touching the rest of the pipeline" idea as
the earlier Claude → Gemini swap, just with a runtime switch instead of
editing the code.
"""

import os


def generate_answer(prompt: str) -> str:
    backend = os.environ.get("GENERATION_BACKEND", "gemini").strip().lower()
    if backend == "bedrock":
        return _generate_bedrock(prompt)
    elif backend == "gemini":
        return _generate_gemini(prompt)
    else:
        raise RuntimeError(
            f"Unknown GENERATION_BACKEND '{backend}' — set it to 'gemini' or 'bedrock' in .env"
        )


def _generate_gemini(prompt: str) -> str:
    from google import genai

    from rag.config import get_config

    # get_config checks GEMINI_API_KEY directly first (the .env/local path);
    # only if that's absent does it look at GEMINI_API_KEY_PARAM and fetch
    # from AWS Parameter Store (the App Runner path). Nothing here changes
    # for anyone still just using a plain .env file.
    api_key = get_config("GEMINI_API_KEY", param_name_env="GEMINI_API_KEY_PARAM")

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def _generate_bedrock(prompt: str) -> str:
    import boto3

    region = os.environ.get("AWS_REGION", "ap-south-2")
    model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-haiku-20241022-v1:0")

    # No explicit AWS keys here on purpose: boto3 automatically picks up
    # credentials from (in order) environment variables, ~/.aws/credentials,
    # or — the recommended path once this runs on EC2 — an IAM role attached
    # to the instance. That means nothing AWS-secret-shaped needs to live in
    # .env at all when deployed.
    client = boto3.client("bedrock-runtime", region_name=region)

    # The Converse API is provider-agnostic: the same call shape works for
    # Claude, Amazon Nova, or any other Bedrock model that supports it —
    # unlike the older invoke_model API, which needs a different request
    # body per model family.
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 500},
    )
    return response["output"]["message"]["content"][0]["text"]

"""
config.py — configuration loader that checks environment variables first,
then falls back to AWS Systems Manager Parameter Store.

This is what lets the exact same code run two ways: locally with a plain
.env file (fast to iterate on), and on App Runner with secrets pulled from
Parameter Store via the instance's IAM role (nothing secret baked into the
Docker image or sitting in an env var visible in the App Runner console).
"""

import os
from functools import lru_cache

import boto3


@lru_cache(maxsize=None)
def get_config(name: str, param_name_env: str | None = None, region: str | None = None) -> str:
    """
    Resolve a config value in two steps:
    1. If `name` is set directly as an environment variable, use it as-is —
       this is the local/.env path (e.g. GEMINI_API_KEY=AIzaSy... in .env).
    2. Otherwise, if `param_name_env` is given and set as an env var, treat
       *its* value as a Parameter Store parameter name and fetch the real
       value from SSM — this is the App Runner path (e.g. an env var
       GEMINI_API_KEY_PARAM=/rag-project/gemini-api-key on the App Runner
       service, resolved here to the actual key).

    Cached with lru_cache so each parameter is only fetched from SSM once
    per process, not on every request.
    """
    value = os.environ.get(name)
    if value:
        return value

    if param_name_env:
        param_name = os.environ.get(param_name_env)
        if param_name:
            resolved_region = region or os.environ.get("AWS_REGION", "ap-south-2")
            ssm = boto3.client("ssm", region_name=resolved_region)
            response = ssm.get_parameter(Name=param_name, WithDecryption=True)
            return response["Parameter"]["Value"]

    raise RuntimeError(
        f"Config value '{name}' not found — set it directly as an env var, "
        f"or set {param_name_env or '<PARAM_NAME_ENV>'} to a Parameter Store path."
    )

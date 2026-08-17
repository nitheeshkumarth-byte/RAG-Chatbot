# Slim base image — we don't need the full Python image's extra tooling
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer from app code) so Docker can
# cache this step and skip re-downloading torch/transformers on every
# rebuild when only your .py files change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code
COPY . .

# Make sure the folder documents get saved into exists inside the image
RUN mkdir -p sample_docs

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# ZEN voice server as a container (used for a Hugging Face Docker Space;
# runs anywhere Docker does). The model is baked in so a cold start only has
# to load it, not download it.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KOKORO_MODEL=/app/models/kokoro-v1.0.onnx \
    KOKORO_VOICES=/app/models/voices-v1.0.bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG MODEL_BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
RUN mkdir models \
    && curl -fsSL -o models/kokoro-v1.0.onnx "$MODEL_BASE/kokoro-v1.0.onnx" \
    && curl -fsSL -o models/voices-v1.0.bin "$MODEL_BASE/voices-v1.0.bin"

COPY server.py .
USER app
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]

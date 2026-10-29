# TerraLens API + web UI on CPU.
#   docker build -t terralens .
#   docker run -p 8000:8000 terralens        # http://localhost:8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch keeps the image around 1 GB instead of ~5 GB with CUDA.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Bake the released weights into the image (checksum-verified). Pass an empty
# MODEL_URL to build without them and mount a checkpoint at runtime instead.
ARG MODEL_URL=https://github.com/Med00amine/TerraLens/releases/download/v1.0.0/terralens-v1.0.0.pt
ENV TERRALENS_MODEL=/app/models/terralens.pt
RUN if [ -n "$MODEL_URL" ]; then terralens download-model --url "$MODEL_URL"; fi

RUN useradd --create-home --uid 1000 terralens && chown -R terralens /app
USER terralens

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["uvicorn", "terralens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

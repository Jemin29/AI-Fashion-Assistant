# =============================================================================
# Stage 1 — Builder Stage: Install build-tools and compile PyTorch CUDA/CPU wheels
# =============================================================================
FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Install compilation prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Setup virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pre-install CUDA-enabled PyTorch & TorchVision (GPU support with CPU fallback capacity)
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# Install requirements from root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install backend dependencies (FastAPI, Celery, Redis, Rate Limits, Telemetry)
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn>=0.22.0 \
    celery>=5.3.0 \
    redis>=4.6.0 \
    slowapi>=0.1.8 \
    pydantic-settings>=2.0.0 \
    pyjwt>=2.7.0 \
    passlib[bcrypt]>=1.7.4 \
    psutil>=5.9.0 \
    gradio>=4.0.0

# Install application package
COPY setup.py .
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# =============================================================================
# Stage 2 — Runner Stage: Hardened, lightweight runtime environment
# =============================================================================
FROM python:3.11-slim as runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Install essential runtime library packages for OpenCV, PIL, and processes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and package files from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Copy execution code modules
COPY apps/ ./apps/
COPY configs/ ./configs/
COPY datasets/ ./datasets/
COPY week6/ ./week6/
COPY week7/ ./week7/
COPY week8/ ./week8/

# Create application workspace directory structure
RUN mkdir -p outputs/generated outputs/recommendations outputs/style_manager outputs/knowledge_base outputs/vector_db outputs/sketches logs

# Expose REST API (8000) and Gradio UI (7860) ports
EXPOSE 8000 7860

# Default execution: Launch Gradio client (Overridable by deployment overrides)
CMD ["python", "apps/gradio_app/app.py"]

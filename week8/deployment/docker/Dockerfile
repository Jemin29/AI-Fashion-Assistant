# =============================================================================
# Stage 1 — Builder stage to compile dependencies and setup python environment
# =============================================================================
FROM python:3.11-slim as builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Install system compiler utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Setup virtual environment to containerize packages cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source packages
COPY setup.py .
COPY src/ ./src/
COPY configs/ ./configs/
COPY datasets/ ./datasets/
COPY apps/ ./apps/
COPY week7/ ./week7/
COPY week6/ ./week6/

# Install the application package
RUN pip install --no-cache-dir -e .

# =============================================================================
# Stage 2 — Runner stage containing final minimal runtime dependencies
# =============================================================================
FROM python:3.11-slim as runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Install OS libraries for OpenCV, Pillow, and Faust integrations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    procps \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment and files from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app

# Create output folders with correct permissions
RUN mkdir -p outputs/generated outputs/recommendations outputs/style_manager outputs/knowledge_base outputs/vector_db outputs/sketches logs

# Expose UI and REST API default ports
EXPOSE 7860 8000

# Default entrypoint launches the Gradio interactive UI
CMD ["python", "apps/gradio_app/app.py"]

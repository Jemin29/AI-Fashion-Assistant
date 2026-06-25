# Use official lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Set working directory in container
WORKDIR /app

# Install system dependencies needed for some Python packages (e.g. build-essential, git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy setup.py and packages
COPY setup.py .
COPY src/ ./src/
COPY apps/ ./apps/
COPY configs/ ./configs/
COPY datasets/ ./datasets/

# Install the project in editable/local mode
RUN pip install --no-cache-dir -e .

# Expose ports: 7860 (Gradio) and 8000 (FastAPI)
EXPOSE 7860 8000

# Default command launches the Gradio UI (can be overridden to launch API)
CMD ["python", "apps/gradio_app/app.py"]

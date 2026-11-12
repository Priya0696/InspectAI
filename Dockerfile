
FROM python:3.11-slim-bookworm

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

# System dependencies for OpenCV, Open3D, PyTorch and Anomalib
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libegl1 \
    libgomp1 \
    libgfortran5 \
    libstdc++6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /usr/local/bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install locked Python dependencies
RUN uv sync --locked --no-dev --no-install-project

# Copy InspectAI application
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# Expose FastAPI port
EXPOSE 8001

# Start FastAPI
CMD ["uv", "run", "--no-sync", "uvicorn", \
     "inspectai.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001"]

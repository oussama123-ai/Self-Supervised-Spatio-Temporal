FROM nvcr.io/nvidia/pytorch:22.12-py3

LABEL maintainer="Oussama El Othmani <oussama.elothmani@ept.u-carthage.tn>"
LABEL description="SSS-TT: Self-Supervised Sequential Spatio-Temporal Transformers for Neonatal Pain Assessment"

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create directories
RUN mkdir -p checkpoints/{mae,sss-tt} results data

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command: run training
CMD ["python", "scripts/train.py", "--help"]

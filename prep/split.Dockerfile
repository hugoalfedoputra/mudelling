FROM python:3.13-slim

# Force Python stdout and stderr streams to be unbuffered
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Use rclone's official install script (Auto-detects amd64 vs arm64)
RUN curl https://rclone.org/install.sh | bash

# Set working directory
WORKDIR /app

# Install python dependencies
COPY split-requirements.txt .
RUN pip install --no-cache-dir -r split-requirements.txt

# Copy the python script
COPY split_remote_melspecs.py .

# Run the script natively
CMD ["python", "split_remote_melspecs.py"]
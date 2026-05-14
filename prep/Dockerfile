FROM python:3.13-slim

# Install system dependencies for librosa and rclone download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install rclone inside the container
RUN curl -O https://downloads.rclone.org/v1.74.1/rclone-v1.74.1-linux-amd64.zip \
    && unzip rclone-v1.74.1-linux-amd64.zip \
    && cd rclone-*-linux-amd64 \
    && cp rclone /usr/bin/ \
    && chown root:root /usr/bin/rclone \
    && chmod 755 /usr/bin/rclone \
    && cd .. \
    && rm -rf rclone-* rclone-v1.74.1-linux-amd64.zip

# Set working directory
WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the python script
COPY preprocess.py .

# Create the temp folder for use by the script
RUN mkdir -p /app/temp

# Run the script
CMD ["python", "preprocess.py"]
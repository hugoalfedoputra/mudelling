FROM python:3.13-slim

# Force Python stdout and stderr streams to be unbuffered so that there's logs in the docker container
ENV PYTHONUNBUFFERED=1

# Install system dependencies for librosa and rclone download
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install rclone inside the container
# RUN curl -O https://downloads.rclone.org/v1.74.1/rclone-v1.74.1-linux-amd64.zip \
#     && unzip rclone-v1.74.1-linux-amd64.zip \
#     && cd rclone-*-linux-amd64 \
#     && cp rclone /usr/bin/ \
#     && chown root:root /usr/bin/rclone \
#     && chmod 755 /usr/bin/rclone \
#     && cd .. \
#     && rm -rf rclone-* rclone-v1.74.1-linux-amd64.zip

RUN curl https://rclone.org/install.sh | bash

# Set working directory
WORKDIR /app

# Install python dependencies
COPY migrate-requirements.txt .
RUN pip install --no-cache-dir -r migrate-requirements.txt

# Copy the python script
COPY migrate_melspecs.py .

# Run the script (unbuffered logs)
CMD ["python", "migrate_melspecs.py"]
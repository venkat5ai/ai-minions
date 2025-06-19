# Stage 1: Builder
# This stage installs build-time dependencies and builds Python packages
FROM python:3.10-slim-bookworm AS builder

# Set the working directory
WORKDIR /app

# Install system dependencies required for PDF processing (poppler-utils)
# and build tools (build-essential, pkg-config).
# Combine commands to reduce layers and clean up apt cache in the same step.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    poppler-utils \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip to the latest version to avoid warnings and potential issues
RUN pip install --upgrade pip

# Install Python dependencies. Use --no-cache-dir and pip cache purge
# to minimize cached data within this layer.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Stage 2: Final Image
# This stage copies only the necessary runtime artifacts from the builder stage
FROM python:3.10-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy only the installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install only runtime system dependencies (poppler-utils for PDF loading)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create directories for temporary document storage and FAISS index persistence in one go
# RUN mkdir -p /app/data /app/faiss_indexes

# Copy the application code
COPY . .

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set environment variables (for Flask dev server, though Gunicorn is used by entrypoint)
# Environment variables
ENV FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=3010 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
    # PYTHONFAULTHANDLER=1

# Removed: ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python - no longer needed with FAISS

# Expose the port the Flask app runs on
EXPOSE 3010

# Set the ENTRYPOINT to our script.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Set the default CMD (arguments for the ENTRYPOINT).
CMD ["cloud"]
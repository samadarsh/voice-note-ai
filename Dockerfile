# VoiceNote AI — requires GROQ_API_KEY at runtime (see docker-compose env_file: .env)
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Whisper (ffmpeg)
# and build tools for any python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ---- Base ----
FROM python:3.10-slim

# Set workdir
WORKDIR /app
ENV TRANSFORMERS_CACHE=/tmp/.cache
RUN mkdir -p /tmp/.cache && chmod -R 777 /tmp/.cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (including data/ and faiss_index/)
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ---- Pre-download MiniLM embeddings at build time ----
# The model will be stored in the default Hugging Face cache (~/.cache/huggingface)
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"


COPY . .

# ---- Copy FAISS index to /tmp at runtime ----
# We'll copy them from /app/faiss_index in CMD, since /tmp is the only writable location in Spaces
# We will do this in an entrypoint script

# Expose port
EXPOSE 8000

# Run entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]


# Hugging Face Space (Docker SDK) — all-in-one, open-source, no API key.
# Runs Ollama + a small open-source model (llama3.2:3b) and the Streamlit app
# in a single container. The embedding model and FAISS index are baked at build
# time; the LLM is pulled on first start.
FROM python:3.11-slim

# System deps + Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run as uid 1000 with a writable home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_MODELS=/home/user/.ollama/models \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_FLASH_ATTENTION=1 \
    LLM_PROVIDER=ollama \
    OLLAMA_BASE_URL=http://localhost:11434 \
    SUMMARY_MODEL=llama3.2:3b \
    AGENT_MODEL=llama3.2:3b

WORKDIR /home/user/app

# Pull the application code. Pin to your repo; change the URL/branch if you fork it.
ARG APP_REPO=https://github.com/Abiramashree/teacher-intelligence-agent.git
ARG APP_REF=main
RUN git clone --depth 1 --branch "$APP_REF" "$APP_REPO" .

RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model + FAISS index into the image so first query is fast.
RUN python scripts/build_index.py

EXPOSE 7860

# Start Ollama, wait for it, pull the model, then launch Streamlit on the HF port.
CMD bash -lc '\
    ollama serve > /tmp/ollama.log 2>&1 & \
    until curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done; \
    ollama pull "$AGENT_MODEL"; \
    streamlit run app.py \
        --server.port=7860 \
        --server.address=0.0.0.0 \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --server.headless=true'

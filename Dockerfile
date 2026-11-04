FROM python:3.11-slim

WORKDIR /app

# System deps for faiss / sentence-transformers wheels are bundled; keep image lean.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Build the index at container start (idempotent), then launch the UI.
CMD ["sh", "-c", "python scripts/build_index.py && streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]

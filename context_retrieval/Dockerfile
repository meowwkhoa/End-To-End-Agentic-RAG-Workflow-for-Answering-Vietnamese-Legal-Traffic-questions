FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8005
ENV WEAVIATE_URL=http://weaviate.weaviate.svc.cluster.local:85
ENV VECTORIZE_URL=http://emb-svc.emb.svc.cluster.local:65001/vectorize

# Expose the port your app runs on
EXPOSE $PORT

# Start the FastAPI app using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8007
ENV RAG_SERVICE_URL=http://rag-agent.rag-agent.svc.cluster.local:65003/process-query

# Expose the port your app runs on
EXPOSE $PORT

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]

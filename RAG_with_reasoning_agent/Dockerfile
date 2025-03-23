FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Expose the port your app runs on
EXPOSE 8006

ENV CONTEXT_SERVICE_URL=http://retrieval.context-retrieval.svc.cluster.local:82/retrieve-context

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]

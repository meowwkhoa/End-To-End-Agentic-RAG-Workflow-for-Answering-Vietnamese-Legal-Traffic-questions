FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8006
ENV CONTEXT_SERVICE_URL=http://retrieval.context-retrieval.svc.cluster.local:65002/retrieve-context

# Expose the port your app runs on
EXPOSE $PORT

# Start the FastAPI app using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]

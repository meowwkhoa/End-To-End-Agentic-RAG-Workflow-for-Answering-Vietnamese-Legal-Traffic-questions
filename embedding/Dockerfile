FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=5000

# Expose the port your app runs on
EXPOSE $PORT

# Start the FastAPI app using uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]

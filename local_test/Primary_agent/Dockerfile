FROM python:3.9-slim

WORKDIR /app

# Copy dependency specifications
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8006

ENV RAG_SERVICE_URL=http://172.17.0.1:8006/process-query

# Start the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8007"]

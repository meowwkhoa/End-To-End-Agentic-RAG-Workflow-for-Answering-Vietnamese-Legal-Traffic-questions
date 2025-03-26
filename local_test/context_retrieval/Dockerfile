FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8005

ENV WEAVIATE_URL=http://172.17.0.1:8080
ENV VECTORIZE_URL=http://172.17.0.1:5000/vectorize

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
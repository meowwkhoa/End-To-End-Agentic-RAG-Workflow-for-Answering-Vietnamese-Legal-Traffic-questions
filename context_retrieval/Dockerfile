FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8005

ENV WEAVIATE_URL=http://weaviate.weaviate.svc.cluster.local:85
ENV VECTORIZE_URL=http://emb-svc.emb.svc.cluster.local:81/vectorize

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
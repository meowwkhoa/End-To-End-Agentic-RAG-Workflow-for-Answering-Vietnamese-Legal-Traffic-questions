from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
import httpx
from pyvi.ViTokenizer import tokenize
import uvicorn
import asyncio
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

# Tracing Setup
JAEGER_HOST = os.getenv("JAEGER_HOST", "jaeger-tracing.jaeger-tracing.svc.cluster.local")
JAEGER_PORT = int(os.getenv("JAEGER_PORT", 6831))

trace_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "Context-Retrieval"}))

jaeger_exporter = JaegerExporter(agent_host_name=JAEGER_HOST, agent_port=JAEGER_PORT)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)

# FastAPI setup with OpenTelemetry
app = FastAPI()
HTTPXClientInstrumentor().instrument() 
FastAPIInstrumentor.instrument_app(app, tracer_provider=trace_provider)


WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate.weaviate.svc.cluster.local:85")
VECTORIZE_URL = os.getenv("VECTORIZE_URL", "http://emb-svc.emb.svc.cluster.local:65001/vectorize")

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    alpha: float = 0.5

async def fetch_vectorized_query(tokenized_query: str):
    """Sends the tokenized query to the embedding service and retrieves vector representation."""
    async with httpx.AsyncClient() as client:
        response = await client.post(VECTORIZE_URL, json={"text": tokenized_query})
        response.raise_for_status()
        return response.json().get("vector")

@app.post("/retrieve-context")
async def retrieve_context(request: QueryRequest):
        client_weaviate = weaviate.Client(url=WEAVIATE_URL)
        tokenized_query = tokenize(request.query)

        # Fetch vector embedding asynchronously
        query_vector = await fetch_vectorized_query(tokenized_query)

        # Query Weaviate database
        res = await asyncio.to_thread(
            lambda: client_weaviate.query.get("Document", ["content"])
                .with_hybrid(query=request.query, alpha=request.alpha, vector=query_vector)
                .with_limit(request.limit)
                .do()
        )

        # Format the retrieved context
        contents = [doc["content"] for doc in res["data"]["Get"]["Document"]]
        context_string = "\n------------------------------------------------------------\n".join(contents)


        return {"context": context_string}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
import requests
from pyvi.ViTokenizer import tokenize
import uvicorn
import asyncio
import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tracing Setup
JAEGER_HOST = os.getenv("JAEGER_HOST", "localhost")
JAEGER_PORT = int(os.getenv("JAEGER_PORT", 6831))
trace_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "Context-Retrieval"}))
tracer = trace_provider.get_tracer("context-retrieval-tracer")
jaeger_exporter = JaegerExporter(agent_host_name=JAEGER_HOST, agent_port=JAEGER_PORT)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
RequestsInstrumentor().instrument()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate.weaviate.svc.cluster.local:85")
VECTORIZE_URL = os.getenv("VECTORIZE_URL", "http://emb-svc.emb.svc.cluster.local:81/vectorize")

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    alpha: float = 0.5

@app.post("/retrieve-context")
async def retrieve_context(request: QueryRequest):
    with tracer.start_as_current_span("retrieve_context") as span:
        client_weaviate = weaviate.Client(url=WEAVIATE_URL)
        tokenized_query = tokenize(request.query)
        text_data = {"text": tokenized_query}
        response = await asyncio.to_thread(requests.post, VECTORIZE_URL, json=text_data)
        response.raise_for_status()
        query_vector = response.json().get("vector")
    
        res = await asyncio.to_thread(
            lambda: client_weaviate.query.get("Document", ["content"])
                .with_hybrid(query=request.query, alpha=request.alpha, vector=query_vector)
                .with_limit(request.limit)
                .do()
        )
    
        contents = [doc["content"] for doc in res["data"]["Get"]["Document"]]
        context_string = "\n------------------------------------------------------------\n".join(contents)
        span.set_attribute("context.length", len(context_string))
        return {"context": context_string}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)

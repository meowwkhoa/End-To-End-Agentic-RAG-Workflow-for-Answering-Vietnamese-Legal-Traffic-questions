from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
import uvicorn
import os
import logging
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

load_dotenv()

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tracing Setup
JAEGER_HOST = os.getenv("JAEGER_HOST", "jaeger-tracing.jaeger-tracing.svc.cluster.local")
JAEGER_PORT = int(os.getenv("JAEGER_PORT", 6831))

trace_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "Primary-Agent"}))
tracer = trace_provider.get_tracer("primary-agent-tracer")
jaeger_exporter = JaegerExporter(agent_host_name=JAEGER_HOST, agent_port=JAEGER_PORT)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)

app = FastAPI()
# Automatic Instrumentation for FastAPI and httpx
HTTPXClientInstrumentor().instrument()
FastAPIInstrumentor.instrument_app(app, tracer_provider=trace_provider)


# Configuration
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-agent.rag-agent.svc.cluster.local:65003/process-query")

client = OpenAI(base_url=RUNPOD_BASE_URL, api_key=RUNPOD_API_KEY)

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    alpha: float = 0.5

class RAGResponse(BaseModel):
    status: str
    reasoning: str
    answer: Optional[str]
    refined_query: Optional[str]
    attempts: int
    response: str

async def call_runpod(prompt: str) -> str:
    """Calls RunPod API to generate a response."""
    try:
            logger.info("Calling RunPod API (Primary Agent)")
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct", 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    top_p=0.8,
                    max_tokens=700,
                )
            )

            result = response.choices[0].message.content
            logger.info("RunPod API call successful")
            return result
    except Exception as e:
            logger.error(f"Error in RunPod API call (Primary Agent): {e}")
            raise HTTPException(status_code=500, detail="RunPod API error")

async def call_rag_service(request: QueryRequest):
    async with httpx.AsyncClient() as client:
        try:
            logger.info("Calling RAG service")
            response = await client.post(RAG_SERVICE_URL, json=request.dict(), timeout=None)
            response.raise_for_status()
            logger.info("Successfully received response from RAG service")
            return response.json()
        except Exception as e:
            logger.error(f"Error calling RAG service: {e}")
            raise HTTPException(status_code=500, detail=f"Error calling RAG service: {str(e)}")

@app.post("/primary-agent", response_model=RAGResponse)
async def primary_agent_endpoint(request: QueryRequest):
        logger.info(f"Received request: {request.query}")
        prompt = f"""
            You are an intelligent AI assistant specialized in answering user queries.

            **Instructions:**
            1. If the user's question is related to Vietnamese traffic law, respond exactly with "USE_RAG".
            2. If the question is trivial, provide a direct answer.
            3. Otherwise, provide a concise and accurate response.

            **User's Question:** "{request.query}"
        """
        response = await call_runpod(prompt)
        response = response.strip()
        
        if "USE_RAG" in response:
            logger.info("Query classified as requiring RAG service")
            rag_response = await call_rag_service(request)
            return rag_response
        else:
            logger.info("Returning direct response")
            return {
                "status": "direct_answer",
                "reasoning": "",
                "answer": response,
                "refined_query": None,
                "attempts": 1,
                "response": response
            }
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)

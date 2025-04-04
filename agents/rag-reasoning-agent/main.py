from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
import uvicorn
import re
import asyncio
from openai import OpenAI
import os
import logging
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
trace_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "RAG-Agent"}))
tracer = trace_provider.get_tracer("rag-agent-tracer")
jaeger_exporter = JaegerExporter(agent_host_name=JAEGER_HOST, agent_port=JAEGER_PORT)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)

app = FastAPI()
HTTPXClientInstrumentor().instrument()
FastAPIInstrumentor.instrument_app(app, tracer_provider=trace_provider)


# Configuration
RUNPOD_ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
RUNPOD_API_KEY = os.getenv('RUNPOD_API_KEY')
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"
CONTEXT_SERVICE_URL = os.getenv("CONTEXT_SERVICE_URL", "http://retrieval.context-retrieval.svc.cluster.local:65002/retrieve-context")

client = OpenAI(base_url=RUNPOD_BASE_URL, api_key=RUNPOD_API_KEY)

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    alpha: float = 0.5

class RAGResponse(BaseModel):
    status: str  # "success", "refined_success", "max_retries_exceeded"
    reasoning: str
    answer: Optional[str]
    refined_query: Optional[str]
    attempts: int
    response: str

async def call_runpod(prompt: str) -> str:
    """Calls RunPod API to generate a response asynchronously."""
    try:
            logger.info("Calling RunPod API (RAG Agent)")
            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    top_p=0.8,
                    max_tokens=700,
                )
            )

            result = response.choices[0].message.content
            logger.info("RunPod API call successful (RAG Agent)")
            return result
    except Exception as e:
            logger.error(f"Error in RunPod API call (RAG Agent): {e}")
            raise HTTPException(status_code=500, detail="RunPod API error in RAG Agent")


async def fetch_context(query, request):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            CONTEXT_SERVICE_URL,
            json={"query": query, "limit": request.limit, "alpha": request.alpha}
        )
        response.raise_for_status()
        return response.json().get("context", "")

def extract_response(response: str) -> tuple:
    """Extract components from model response without fallback default strings."""
    parts = response.split("</think>", 1)
    reasoning = parts[0].strip()
    answer = parts[1].strip() if len(parts) > 1 else ""
    
    refined_query_match = re.search(r'Refined Query:\s*(.+)', answer, re.IGNORECASE)
    refined_query = refined_query_match.group(1).strip() if refined_query_match else None

    if refined_query:
        return reasoning, "", refined_query
    
    return reasoning, answer, None

@app.post("/process-query", response_model=RAGResponse)
async def process_query(request: QueryRequest):
        original_query = request.query
        query = original_query
        max_retries = 2  # Allow one refine attempt (total attempts = 2)
        attempt_logs = []
    
        for attempt in range(max_retries):
                try:
                    # Get context from the context service
                    context = await fetch_context(query, request)
    
                    # Generate prompt using the provided context and query
                    prompt = f"""
                        You are an expert in Vietnamese Traffic Law. Your task is to analyze the user's question using the provided context and respond exclusively in Vietnamese.
    
                        Take as much time as you need to study the problem carefully and methodically—there is no rush for an immediate answer.
    
                        **Instructions:**
                        1. **Assess Context Sufficiency:**
                        - If the context is sufficient to answer the question, provide a clear, step-by-step reasoning process followed by a direct answer.
                        - If the context is insufficient, explain why and propose a refined query using the exact format:
                            Final Output: "Refined Query: <new query in Vietnamese>"
                        2. **Response Format:**
                        - Your answer must strictly follow this structure:
                            <think> [Your reasoning process in Vietnamese] </think>
                            <answer> [Your final answer in Vietnamese] OR [Refined Query: <new query in Vietnamese>] </answer>
                        3. **Language Requirement:**
                        - Respond only in Vietnamese.
    
                        **Input:**
                        - **Context:** {context}
                        - **Question:** {query}
    
                        Now, analyze the problem and respond in the required format.
                    """
    
                    # Get model response from RunPod
                    model_response = await call_runpod(prompt)
                    reasoning, answer, refined_query = extract_response(model_response)
    
                    attempt_logs.append({
                        "attempt": attempt + 1,
                        "query_used": query,
                        "reasoning": reasoning,
                        "answer": answer,
                        "refined_query": refined_query
                    })
    
                    if answer.strip():
                        break
    
                    if refined_query:
                        query = refined_query
                    else:
                        break
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
    
        final_attempt = attempt_logs[-1]
        status = "success" if final_attempt['attempt'] == 1 else "refined_success" if final_attempt['answer'].strip() else "max_retries_exceeded"
    
        return RAGResponse(
            status=status,
            reasoning=final_attempt['reasoning'],
            answer=final_attempt['answer'] if final_attempt['answer'].strip() else None,
            refined_query=final_attempt['refined_query'],
            attempts=len(attempt_logs),
            response=str(attempt_logs)
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)

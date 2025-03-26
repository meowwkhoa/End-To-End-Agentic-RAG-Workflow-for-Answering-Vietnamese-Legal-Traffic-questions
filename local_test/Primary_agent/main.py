from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import uvicorn
from openai import OpenAI

app = FastAPI()

# Configuration
RUNPOD_ENDPOINT_ID = "6r7e7wvauz0a8m"
RUNPOD_API_KEY = "rpa_999TO9T6WP6ZQGCPPA83HDHJUMLI9QFZ2PTPC1F2fmee68"
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
RAG_SERVICE_URL = "http://172.17.0.1:8006/process-query"  # URL for the RAG service

client = OpenAI(
    base_url=RUNPOD_BASE_URL,
    api_key=RUNPOD_API_KEY,
)

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    alpha: float = 0.5

class RAGResponse(BaseModel):
    status: str  # "direct_answer", "rag_success", "rag_refined_success", "rag_max_retries_exceeded"
    reasoning: str
    answer: Optional[str]
    refined_query: Optional[str]
    attempts: int
    response: str  # Combined detailed log or the direct answer

def call_runpod(prompt: str) -> str:
    """
    Calls RunPod API to generate a response.
    """
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.8,
        max_tokens=700,
    )
    return response.choices[0].message.content


def call_ollama(prompt: str) -> str:
    payload = {
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "top_p": 0.8,
            "max_tokens": 500,
        },
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

@app.post("/primary-agent", response_model=RAGResponse)
async def primary_agent_endpoint(request: QueryRequest):
    # Create a prompt to decide if the question is related to Vietnamese traffic law.
    prompt = f"""
        You are an intelligent AI assistant specialized in answering user queries.

        **Instructions:**
        1. If the user's question is related to Vietnamese traffic law (e.g., regulations, fines, violations, legal procedures), respond exactly with "USE_RAG".
        2. If the user's question is trivial or not related to Vietnamese traffic law (for example, greetings like "hello" or "hi"), provide a simple direct answer.
        3. Otherwise, if you can confidently answer the question directly without additional context, provide a concise and accurate response.

        **User's Question:**
        "{request.query}"
    """
    response = call_ollama(prompt).strip()
    # If the response instructs to use RAG, call the RAG service.
    if "USE_RAG" in response:
        try:
            rag_response = requests.post(RAG_SERVICE_URL, json=request.dict())
            rag_response.raise_for_status()
            return rag_response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling RAG service: {str(e)}")
    else:
        # Return direct answer with same output format.
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

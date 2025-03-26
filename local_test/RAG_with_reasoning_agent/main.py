from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import uvicorn
import re
from openai import OpenAI

app = FastAPI()

RUNPOD_ENDPOINT_ID = "6r7e7wvauz0a8m"
RUNPOD_API_KEY = "rpa_999TO9T6WP6ZQGCPPA83HDHJUMLI9QFZ2PTPC1F2fmee68"
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"

# Configuration
CONTEXT_SERVICE_URL = "http://172.17.0.1:8005/retrieve-context"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

client = OpenAI(
    base_url=RUNPOD_BASE_URL,
    api_key=RUNPOD_API_KEY,
)

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
    response: str  # For backward compatibility

def call_runpod(prompt: str) -> str:
    """
    Calls RunPod API to generate a response.
    """
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",  # Adjust to your chosen model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.8,
        max_tokens=700,
    )
    return response.choices[0].message.content

def call_ollama(prompt: str) -> str:
    """Helper function to call Ollama API"""
    payload = {
        "model": "deepseek-r1:7b",
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

def extract_response(response: str) -> tuple:
    """Extract components from model response without fallback default strings."""
    parts = response.split("</think>", 1)
    reasoning = parts[0].strip()
    answer = parts[1].strip() if len(parts) > 1 else ""
    
    # Look for a refined query in the answer part
    refined_query_match = re.search(r'Refined Query:\s*(.+)', answer, re.IGNORECASE)
    refined_query = refined_query_match.group(1).strip() if refined_query_match else None

    # If a refined query is detected, we keep the answer as empty.
    if refined_query:
        return reasoning, "", refined_query
    
    return reasoning, answer, None

@app.post("/process-query", response_model=RAGResponse)
async def process_query(request: QueryRequest):
    """
    Main endpoint for the RAG reasoning process.
    
    This function handles three cases:
      1. Direct answer on the first attempt.
      2. A refined query is generated and, on the second attempt, an answer is provided.
      3. A refined query is generated but even after refinement no answer is obtained.
    
    All details (reasoning, answer, refined query) from each attempt are included in the final output.
    """
    original_query = request.query
    query = original_query
    max_retries = 2  # Allow one refine attempt (total attempts = 2)
    attempt_logs = []
    
    for attempt in range(max_retries):
        try:
            # Get context from the context service
            context_response = requests.post(
                CONTEXT_SERVICE_URL,
                json={"query": query, "limit": request.limit, "alpha": request.alpha}
            )
            context_response.raise_for_status()
            context = context_response.json().get("context", "")
            
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
                - Respond only in Vietnamese. Do not use any other language (including Chinese) and do not include extra sections.

                **Input:**
                - **Context:** {context}
                - **Question:** {query}

                Now, analyze the problem and respond in the required format.
            """

            
            # Get model response from Ollama
            model_response = call_ollama(prompt)
            reasoning, answer, refined_query = extract_response(model_response)
            
            # Record attempt details
            attempt_logs.append({
                "attempt": attempt + 1,
                "query_used": query,
                "reasoning": reasoning,
                "answer": answer,
                "refined_query": refined_query
            })
            
            # If we get a valid answer (non-empty and not the default placeholder), break.
            if answer is not None and answer.strip():
                break
            
            # If a refined query is provided, update the query and continue to next attempt.
            if refined_query:
                query = refined_query
            else:
                break
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Build a detailed log of all attempts
    detailed_log = "\n\n".join(
        [f"Attempt {log['attempt']} (Query: {log['query_used']}):\nReasoning: {log['reasoning']}\nAnswer: {log['answer']}\nRefined Query: {log['refined_query']}"
         for log in attempt_logs]
    )
    
    final_attempt = attempt_logs[-1]
    
    # Determine final status based on the attempts:
    if final_attempt['answer'] is not None and final_attempt['answer'].strip():
        if final_attempt['attempt'] == 1:
            status = "success"
        else:
            status = "refined_success"
    else:
        status = "max_retries_exceeded"
    
    return RAGResponse(
        status=status,
        reasoning=final_attempt['reasoning'],
        answer=final_attempt['answer'] if final_attempt['answer'] and final_attempt['answer'].strip() else None,
        refined_query=final_attempt['refined_query'],
        attempts=len(attempt_logs),
        response=detailed_log
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)

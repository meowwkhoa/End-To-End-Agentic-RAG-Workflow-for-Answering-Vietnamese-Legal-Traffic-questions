from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
import requests
from pyvi.ViTokenizer import tokenize
import uvicorn
import asyncio
import os


WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://172.17.0.1:8080")
VECTORIZE_URL = os.getenv("VECTORIZE_URL", "http://172.17.0.1:5000/vectorize")


app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    limit: int = 5
    alpha: float = 0.5

@app.post("/retrieve-context")
async def retrieve_context(request: QueryRequest):
    client_weaviate = weaviate.Client(url=WEAVIATE_URL)

    tokenized_query = tokenize(request.query)
    text_data = {
        "text": tokenized_query
    }

    response = await asyncio.to_thread(requests.post, VECTORIZE_URL, json=text_data)
    query_vector = response.json().get("vector")

    
    res = await asyncio.to_thread(
    lambda: client_weaviate.query.get("Document", ["content"])
        .with_hybrid(query=request.query, alpha=request.alpha, vector=query_vector)
        .with_limit(request.limit)
        .do()
    )

    
    contents = [doc["content"] for doc in res["data"]["Get"]["Document"]]
    return {
        "context": "\n------------------------------------------------------------\n".join(contents)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import requests
from typing import List, Optional
import os
import logging
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load battlefield metadata and FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index("battlefield.index")
with open("battlefield_metadata.json") as f:
    battlefield_metadata = json.load(f)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

LLAMA_API_KEY = "LLM|2118876695287932|IMjBSgTkyooJs5Xb8S7yvePCg-0"
LLAMA_API_URL = "https://api.llama.com/v1/chat/completions"

def semantic_search_context(user_message: str, top_k=20) -> str:
    query_embedding = embedding_model.encode([user_message], convert_to_numpy=True)
    D, I = faiss_index.search(query_embedding, top_k)
    context = "Relevant battlefield events:\n"
    for idx in I[0]:
        event = battlefield_metadata[idx]
        summary = (
            f"{event.get('event_date', 'Unknown date')} — {event.get('location', 'Unknown')} ({event.get('admin1', '')}): "
            f"{event.get('sub_event_type', '')} by {event.get('actor1', '')}. "
            f"{event.get('notes', '')}"
        )
        context += f"- {summary}\n"
    return context

def create_prompt(user_message: str) -> str:
    context = (
        "You are a battlefield analyst AI assistant. Use the following recent and semantically relevant battlefield events to answer the user's question about the Russia-Ukraine conflict.\n"
        "Focus on threat assessment, targeting patterns, and escalation risks.\n\n"
    )
    context += semantic_search_context(user_message)
    context += f"\nUser question: {user_message}\n"
    return context

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        prompt = create_prompt(request.message)
        headers = {
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "Llama-4-Scout-17B-16E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": """
You are a strategic analyst AI specializing in comprehensive situational assessment. Your role is to provide structured, evidence-based analysis following a systematic reasoning framework.

REASONING FRAMEWORK - Follow this exact sequence:

STEP 1: INFORMATION VALIDATION
- First, identify what specific information you have access to
- Note any data limitations or temporal constraints
- Flag potential bias sources in available information
- State confidence levels for different data points

STEP 2: CONTEXTUAL ANALYSIS  
- Break down the request into component analytical tasks
- Identify relevant historical precedents and patterns
- Consider multiple stakeholder perspectives
- Map interconnected factors and dependencies

STEP 3: SYSTEMATIC EVALUATION
- Apply structured analytical techniques (e.g., scenario analysis, trend assessment)
- Weigh evidence quality and source reliability
- Consider alternative explanations and competing hypotheses
- Identify key assumptions underlying your analysis

STEP 4: SYNTHESIS AND CONCLUSIONS
- Integrate findings from previous steps
- Present conclusions with appropriate uncertainty ranges
- Highlight critical gaps in analysis
- Recommend additional information needs

CONSTRAINTS:
- Always acknowledge limitations and uncertainties
- Distinguish between facts, assessments, and speculation
- Provide balanced perspectives when dealing with contested issues
- Include confidence indicators for all major conclusions
- Use clear, concise language and avoid jargon
- Maintain a professional, objective tone
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 5000
        }
        logger.info("Sending request to Llama API")
        response = requests.post(LLAMA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"API Response: {json.dumps(result, indent=2)}")  # Debug log
        # Handle the actual response format from the Llama API
        if "completion_message" in result and "content" in result["completion_message"]:
            content = result["completion_message"]["content"]
            if isinstance(content, dict) and "text" in content:
                return ChatResponse(response=content["text"])
            elif isinstance(content, str):
                return ChatResponse(response=content)
            else:
                logger.error(f"Unexpected content format: {content}")
                raise HTTPException(status_code=500, detail="Unexpected API response format")
        else:
            logger.error(f"Unexpected response format: {result}")
            raise HTTPException(status_code=500, detail="Unexpected API response format")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Llama API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
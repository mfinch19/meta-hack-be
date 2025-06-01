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
from fastapi.responses import StreamingResponse
import traceback
from datetime import datetime, timedelta

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

NYT_API_KEY = "inCve8zCkZsR3AGsh82xQrkTJN6Yd3zJ"
NYT_SEARCH_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

NYT_QUERIES = [
    "Ukraine Russia",
    "drone strikes Ukraine",
    "Russia shelling civilian",
    "frontline Ukraine Donbas",
    "NATO support Ukraine",
    "Black Sea Fleet Ukraine",
    "Belgorod border attacks",
    "missile attacks Kyiv",
    "Ukrainian counteroffensive"
]

def extract_date_range(user_message):
    # Try to extract YYYY-MM-DD or YYYY/MM/DD
    date_matches = re.findall(r'(\d{4}[-/]\d{2}[-/]\d{2})', user_message)
    if date_matches:
        # If one date, use as both begin and end; if two, use as range
        if len(date_matches) == 1:
            return date_matches[0].replace('/', ''), date_matches[0].replace('/', '')
        else:
            return date_matches[0].replace('/', ''), date_matches[1].replace('/', '')
    # Heuristic for relative dates
    now = datetime.utcnow()
    if 'yesterday' in user_message.lower():
        day = now - timedelta(days=1)
        return day.strftime('%Y%m%d'), day.strftime('%Y%m%d')
    if 'last week' in user_message.lower():
        start = now - timedelta(days=7)
        return start.strftime('%Y%m%d'), now.strftime('%Y%m%d')
    if 'last month' in user_message.lower():
        start = now - timedelta(days=30)
        return start.strftime('%Y%m%d'), now.strftime('%Y%m%d')
    # Default: last 7 days
    start = now - timedelta(days=7)
    return start.strftime('%Y%m%d'), now.strftime('%Y%m%d')

def fetch_nyt_news(user_message, num_articles=5):
    begin_date, end_date = extract_date_range(user_message)
    query = "Ukraine Russia war"  # More specific and stable

    params = {
        "q": query,
        "sort": "newest",
        "begin_date": begin_date,
        "end_date": end_date,
        "api-key": NYT_API_KEY
    }

    try:
        logger.info(f"Fetching NYT news with params: {params}")
        resp = requests.get(NYT_SEARCH_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        # Log the full response for debugging
        logger.info(f"NYT API Response: {json.dumps(data, indent=2)}")

        # Validate structure
        if not data or "response" not in data or "docs" not in data["response"]:
            logger.error("NYT API returned unexpected structure")
            return "NYT news temporarily unavailable due to an unexpected structure."

        articles = data["response"]["docs"]
        if not articles:
            logger.info(f"No NYT articles found for query: '{query}' between {begin_date} and {end_date}")
            return "No recent NYT news available for the specified time period."

        # Build result string
        news_snippets = []
        for article in articles[:num_articles]:
            headline = article.get("headline", {}).get("main", "No headline")
            snippet = article.get("snippet", "No summary available.")
            pub_date = article.get("pub_date", "")[:10]
            url = article.get("web_url", "#")
            news_snippets.append(f"- [{headline}]({url}) ({pub_date}): {snippet}")
            logger.info(f"Processed article: {headline}")

        result = "\n".join(news_snippets)
        logger.info(f"Final news snippets:\n{result}")
        return result

    except requests.exceptions.Timeout:
        logger.error("NYT API request timed out")
        return "NYT news temporarily unavailable due to timeout."
    except requests.exceptions.RequestException as e:
        logger.error(f"NYT API request error: {e}")
        return "NYT news temporarily unavailable due to request failure."
    except Exception as e:
        logger.error(f"Unexpected NYT fetch error: {e}")
        return "NYT news temporarily unavailable due to internal error."

def semantic_search_context(user_message: str, top_k=7) -> str:
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
    # Add NYT news section with date range relevance
    context += "\nRecent New York Times headlines about the conflict (date range auto-selected for relevance):\n"
    context += fetch_nyt_news(user_message)
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

REASONING FRAMEWORK - Follow this exact sequence and format:

STEP 1: INFORMATION VALIDATION
[In 1-2 sentences, briefly state what information you have, any data limitations, and your confidence level.]

STEP 2: CONTEXTUAL ANALYSIS  
[In 1-2 sentences, briefly analyze relevant patterns, perspectives, and dependencies.]

STEP 3: SYSTEMATIC EVALUATION
[In 1-2 sentences, briefly weigh evidence, consider alternatives, and state key assumptions.]

STEP 4: SYNTHESIS AND CONCLUSIONS
[In 1-2 sentences, briefly integrate findings.]

FINAL ANSWER:
[In 2 sentences, summarize your key findings and recommendations.]

CONSTRAINTS:
- Keep each step to 1-2 sentences maximum
- Keep the final answer to 2 sentences maximum
- Use clear, professional language
- Include confidence levels for major conclusions
- Acknowledge limitations and uncertainties
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 3000,
            "stream": False
        }
        logger.info("Sending request to Llama API (non-streaming)")
        logger.debug(f"Request payload: {json.dumps(payload, indent=2)[:1000]}")
        response = requests.post(LLAMA_API_URL, headers=headers, json=payload)
        logger.info(f"Llama API response status: {response.status_code}")
        logger.debug(f"Llama API response headers: {response.headers}")
        response.raise_for_status()
        result = response.json()
        logger.debug(f"Llama API response JSON: {json.dumps(result, indent=2)[:2000]}")
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
        logger.error(f"API request error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error communicating with Llama API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
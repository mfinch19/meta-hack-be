from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import ollama
from typing import List, Optional
import os
import logging
import re
import faiss
import numpy as np
from llama_api_client import LlamaAPIClient

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

# Get API key from environment variable
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    raise ValueError("LLAMA_API_KEY environment variable is not set")

# Load battlefield metadata and FAISS index
faiss_index = faiss.read_index("battlefield.index")
with open("battlefield_metadata.json") as f:
    battlefield_metadata = json.load(f)

# Initialize Ollama client with remote host
ollama_client = ollama.Client(host="http://3.238.200.222:11434")

def get_embedding(text: str) -> np.ndarray:
    response = ollama_client.embed(
        model="mxbai-embed-large:latest",
        input=text
    )
    return np.array(response['embedding'])

def semantic_search_context(user_message: str, top_k=20) -> str:
    query_embedding = get_embedding(user_message)
    # Ensure the embedding is 2D for FAISS
    query_embedding = query_embedding.reshape(1, -1)
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

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Initialize Llama API client with API key
client = LlamaAPIClient(api_key=LLAMA_API_KEY)

# def create_prompt(user_message: str) -> str:
#     context = (
#         "You are a battlefield analyst AI assistant. Use the following recent and semantically relevant battlefield events to answer the user's question about the Russia-Ukraine conflict.\n"
#         "Focus on threat assessment, targeting patterns, and escalation risks.\n\n"
#     )
#     context += semantic_search_context(user_message)
#     context += f"\nUser question: {user_message}\n"
#     return context


# def create_prompt(user_message: str) -> str:
#     user_message += f"\nUser question: {user_message}\n"
#     return user_message

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # First, extract keywords from the user's message
        keyword_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
You are an elite military intelligence keyword extraction specialist. Extract critical search terms from the provided text for threat assessment and intelligence retrieval systems.

EXTRACT these keyword types with precision:
- Threat entities (organizations, individuals, weapons systems)
- Geographic locations (coordinates, regions, facilities, borders)
- Temporal indicators (dates, timeframes, operational windows)
- Military assets (equipment, vehicles, personnel classifications)
- Operational terms (tactics, procedures, mission types)
- Intelligence classifications (threat levels, capabilities, intentions)
- Technical specifications (ranges, frequencies, capabilities)

PRIORITIZE keywords that enable:
- Rapid threat identification and correlation
- Cross-reference with intelligence databases
- Pattern recognition across multiple sources
- Real-time situational awareness updates

REQUIREMENTS:
- Extract 8-15 keywords maximum
- Include single words AND multi-word phrases
- Rank by operational criticality (most critical first)
- Focus on actionable intelligence terms
- Exclude common military jargon unless contextually critical

FORMAT: Return as comma-separated list, highest priority first.
"""
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            model="Cerebras-Llama-4-Maverick-17B-128E-Instruct",
            stream=False,
            temperature=0.3,
            max_completion_tokens=100
        )
        
        # Extract keywords from response
        keywords = ""
        if hasattr(keyword_response, 'completion_message') and hasattr(keyword_response.completion_message, 'content'):
            content = keyword_response.completion_message.content
            if hasattr(content, 'text'):
                keywords = content.text.strip()
            else:
                logger.warning("Unexpected keyword response format")
        
        # Enhance the original prompt with keywords for better RAG matching
        enhanced_prompt = f"Original query: {request.message}\nRelevant keywords: {keywords}\n"
        
        # Now process the enhanced prompt with the original logic
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
You are a military strategist and analyst AI specializing in comprehensive situational assessment. Your role is to provide structured, evidence-based analysis following a systematic reasoning framework.

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
                    "content": enhanced_prompt
                }
            ],
            model="Cerebras-Llama-4-Maverick-17B-128E-Instruct",
            stream=False,
            temperature=0.4,
            max_completion_tokens=2048,
            top_p=0.9,
            repetition_penalty=1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "general_explanation": {
                                "type": "string",
                                "description": "Overall explanation or summary"
                            },
                            "locations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Name of the city or town"
                                        },
                                        "explanation": {
                                            "type": "string",
                                            "description": "Short reason for selecting this location"
                                        }
                                    },
                                    "required": ["name", "explanation"],
                                    "additionalProperties": False
                                },
                                "description": "Array of location objects with individual analysis for each location"
                            }
                        },
                        "required": ["general_explanation", "locations"],
                        "additionalProperties": False
                    }
                }
            }
        )
        
        # Extract the response content
        if hasattr(response, 'completion_message') and hasattr(response.completion_message, 'content'):
            content = response.completion_message.content
            if hasattr(content, 'text'):
                return ChatResponse(response=content.text)
            else:
                logger.error(f"Unexpected content format: {content}")
                raise HTTPException(status_code=500, detail="Unexpected content format in API response")
        else:
            logger.error(f"Unexpected response format: {response}")
            raise HTTPException(status_code=500, detail="Unexpected API response format")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
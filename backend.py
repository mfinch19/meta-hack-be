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
from datetime import datetime, timedelta
import time
import requests

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
    "Ukrainian counteroffensive",
]


def extract_date_range(user_message):
    # Try to extract YYYY-MM-DD or YYYY/MM/DD
    date_matches = re.findall(r"(\d{4}[-/]\d{2}[-/]\d{2})", user_message)
    if date_matches:
        # If one date, use as both begin and end; if two, use as range
        if len(date_matches) == 1:
            return date_matches[0].replace("/", ""), date_matches[0].replace("/", "")
        else:
            return date_matches[0].replace("/", ""), date_matches[1].replace("/", "")
    # Heuristic for relative dates
    now = datetime.utcnow()
    if "yesterday" in user_message.lower():
        day = now - timedelta(days=1)
        return day.strftime("%Y%m%d"), day.strftime("%Y%m%d")
    if "last week" in user_message.lower():
        start = now - timedelta(days=7)
        return start.strftime("%Y%m%d"), now.strftime("%Y%m%d")
    if "last month" in user_message.lower():
        start = now - timedelta(days=30)
        return start.strftime("%Y%m%d"), now.strftime("%Y%m%d")
    # Default: last 7 days
    start = now - timedelta(days=7)
    return start.strftime("%Y%m%d"), now.strftime("%Y%m%d")


def fetch_nyt_news(user_message, num_articles=5):
    begin_date, end_date = extract_date_range(user_message)
    query = "Ukraine Russia war"  # More specific and stable

    params = {
        "q": query,
        "sort": "newest",
        "begin_date": begin_date,
        "end_date": end_date,
        "api-key": NYT_API_KEY,
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
            logger.info(
                f"No NYT articles found for query: '{query}' between {begin_date} and {end_date}"
            )
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

# Load battlefield metadata and FAISS index
# faiss_index = faiss.read_index("battlefield.index")
faiss_index = faiss.read_index("battlefield.index")
with open("battlefield_metadata.json") as f:
    battlefield_metadata = json.load(f)

# Initialize Ollama client with remote host
try:
    # Try remote host first
    ollama_client = ollama.Client(host="http://3.238.200.222:11434")
    # Test connection
    ollama_client.embeddings(model="mxbai-embed-large:latest", prompt="test")
    logger.info("Successfully connected to remote Ollama server")
except Exception as e:
    logger.warning(f"Failed to connect to remote Ollama server: {str(e)}")
    logger.info("Falling back to local Ollama server")
    ollama_client = ollama.Client()

def get_embedding(text: str) -> np.ndarray:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ollama_client.embed(
                model="mxbai-embed-large:latest",
                input=text
            )
            if 'embeddings' in response:
                return np.array(response['embeddings'])
            else:
                logger.error(f"Unexpected response format: {response}")
                raise ValueError("No embeddings in response")
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"Failed to get embedding after {max_retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            time.sleep(1)  # Wait a second before retrying

def semantic_search_context(user_message: str, top_k=20) -> str:
    try:
        query_embedding = get_embedding(user_message)
        # Log embedding dimensions for debugging
        logger.info(f"Query embedding shape: {query_embedding.shape}")
        logger.info(f"FAISS index dimension: {faiss_index.d}")
        
        # Ensure the embedding is 2D and has the correct dimension
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.shape[1] != faiss_index.d:
            error_msg = f"Embedding dimension mismatch. Expected {faiss_index.d}, got {query_embedding.shape[1]}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        D, I = faiss_index.search(query_embedding, min(top_k, len(battlefield_metadata)))
        logger.info(f"Search completed. Found {len(I[0])} matches")
        
        context = "Relevant battlefield events:\n"
        for idx in I[0]:
            if idx < 0 or idx >= len(battlefield_metadata):
                logger.warning(f"Invalid index {idx} found in search results")
                continue
            event = battlefield_metadata[idx]
            summary = (
                f"{event.get('event_date', 'Unknown date')} — {event.get('location', 'Unknown')} ({event.get('admin1', '')}): "
                f"{event.get('sub_event_type', '')} by {event.get('actor1', '')}. "
                f"{event.get('notes', '')}"
            )
            context += f"- {summary}\n"
        return context
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        # Return a minimal context in case of error to allow the system to continue
        return "Error retrieving battlefield events. Proceeding with limited context.\n"

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Initialize Llama API client with API key
client = LlamaAPIClient(api_key=LLAMA_API_KEY)

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
        # First, extract keywords from the user's message
        logger.info("Sending request for keyword extraction")
        prompt = create_prompt(request.message)
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
                    "content": create_prompt(request.message)
                }
            ],
            model="Cerebras-Llama-4-Maverick-17B-128E-Instruct",
            stream=False,
            temperature=0.3,
            max_completion_tokens=100
        )

        
        # Extract keywords from response
        keywords = ""
        logger.info("Processing keyword response")
        if hasattr(keyword_response, 'completion_message') and hasattr(keyword_response.completion_message, 'content'):
            content = keyword_response.completion_message.content
            if hasattr(content, 'text'):
                keywords = content.text.strip()
            else:
                logger.warning("Unexpected keyword response format: content does not have 'text' attribute")
        else:
            logger.warning(f"Unexpected keyword response format: {dir(keyword_response)}")
        
        # Enhance the original prompt with keywords for better RAG matching
        enhanced_prompt = f"Original query: {prompt}\nRelevant keywords: {keywords}\nCurrent Date and Time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        
        # Generate semantic search context
        logger.info("Generating semantic search context")
        try:
            search_context = semantic_search_context(request.message)
            logger.info("Successfully generated semantic search context")
        except Exception as e:
            logger.error(f"Error generating semantic search context: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating semantic search context: {str(e)}")
        
        # Combine enhanced prompt with search context
        print(search_context)
        full_prompt = enhanced_prompt + search_context
        logger.info("Combined enhanced prompt with search context")
        
        # Now process the enhanced prompt with the original logic
        logger.info("Sending request for main analysis")
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
You are an elite military intelligence analyst with specialized expertise in threat assessment and force protection. Execute comprehensive threat evaluation using this mandatory reasoning framework.

MISSION: Conduct systematic threat assessment to identify high-risk targets and establish safe zone classifications based on available intelligence data.

MANDATORY REASONING SEQUENCE:

STEP 1: INTELLIGENCE VALIDATION & TEMPORAL ANALYSIS
- Validate data sources: Classify information by reliability (A-F scale) and recency
- Temporal correlation: Cross-reference current date/time (UTC) with historical attack patterns
- Data gaps identification: Explicitly state missing critical intelligence
- Source credibility assessment: Weight intelligence based on collection method and verification status

STEP 2: THREAT PATTERN ANALYSIS
- Historical attack vector mapping: Identify recurring tactics, techniques, and procedures (TTPs)
- Geographic correlation analysis: Map attack frequency by location, infrastructure type, and temporal patterns
- Target selection methodology: Analyze adversary target prioritization based on strategic value
- Operational environment assessment: Evaluate terrain, population density, and defensive capabilities

STEP 3: RISK STRATIFICATION FRAMEWORK
- Critical asset vulnerability assessment: Evaluate target hardening, accessibility, and symbolic value
- Threat actor capability matching: Align known adversary capabilities with potential target vulnerabilities
- Probability-impact matrix: Calculate risk scores using standardized military risk assessment protocols
- Cascading effects analysis: Assess secondary and tertiary impacts of potential attacks

STEP 4: SAFE ZONE CLASSIFICATION
- Defensive posture evaluation: Assess force protection measures, early warning systems, and response capabilities
- Geographic advantage analysis: Evaluate natural barriers, controlled access points, and surveillance coverage
- Population protection factors: Consider civilian density, evacuation routes, and medical response capacity
- Intelligence coverage assessment: Evaluate human intelligence (HUMINT) and signals intelligence (SIGINT) penetration

STEP 5: ACTIONABLE INTELLIGENCE SYNTHESIS
- Priority target ranking: List high-risk targets with specific threat timelines and confidence levels
- Safe zone recommendations: Classify areas by security level (Green/Yellow/Orange/Red zones)
- Force protection recommendations: Specify required security measures and resource allocation
- Intelligence collection priorities: Identify critical information requirements for ongoing assessment

OUTPUT REQUIREMENTS:
- Use NATO threat assessment terminology and classification standards
- Include confidence percentages for all major assessments (High: 80-100%, Medium: 50-79%, Low: <50%)
- Specify temporal validity of assessments (e.g., "Valid for 72 hours pending new intelligence")
- Provide clear risk mitigation recommendations for each identified threat
- Format outputs for immediate operational use by command elements

OPERATIONAL CONSTRAINTS:
- Maintain OPSEC protocols: Avoid revealing specific intelligence sources or methods
- Apply appropriate classification handling: Mark sensitive assessments accordingly
- Consider coalition partner equities: Account for multinational force coordination requirements
- Integrate rules of engagement (ROE): Ensure recommendations align with current operational authorities

QUALITY CONTROL:
- Red team analysis: Consider alternative threat scenarios and adversary deception
- Assumption validation: Explicitly state and challenge underlying analytical assumptions
- Uncertainty quantification: Use confidence intervals for numerical assessments
- Bias mitigation: Account for confirmation bias and mirror imaging in analysis

Execute this framework systematically. Begin analysis now.
"""
                    },
                    {
                        "role": "user",
                        "content": full_prompt
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
            print(response)
        except Exception as e:
            logger.error(f"Error in main analysis request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in main analysis request: {str(e)}")
        
        # Extract the response content
        logger.info("Processing main analysis response")
        if hasattr(response, 'completion_message') and hasattr(response.completion_message, 'content'):
            content = response.completion_message.content
            if hasattr(content, 'text'):
                logger.info("Successfully extracted response text")
                return ChatResponse(response=content.text)
            else:
                logger.error(f"Unexpected content format: {content}")
                raise HTTPException(status_code=500, detail="Unexpected content format in API response")
        else:
            logger.error(f"Unexpected response format: {dir(response)}")
            raise HTTPException(status_code=500, detail="Unexpected API response format")
            
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
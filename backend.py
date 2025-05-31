from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import requests
from typing import List, Optional
import os
import logging

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

# Load the battlefield data
def load_battlefield_data():
    try:
        with open('2024-05-24-2025-05-31-Russia-Ukraine.json', 'r') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} battlefield events")
            return data
    except Exception as e:
        logger.error(f"Error loading battlefield data: {str(e)}")
        return []

battlefield_data = load_battlefield_data()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

LLAMA_API_KEY = "LLM|2118876695287932|IMjBSgTkyooJs5Xb8S7yvePCg-0"
LLAMA_API_URL = "https://api.llama.com/v1/chat/completions"

def create_prompt(user_message: str) -> str:
    try:
        # Create a context-aware prompt using the battlefield data
        context = "You are a battlefield analyst AI assistant. Use the following data to provide accurate and insightful responses about the Russia-Ukraine conflict. Focus on threat assessment, target prediction, and strategic analysis.\n\n"
        
        # Get recent events and group them by location
        recent_events = battlefield_data[-20:]  # Get last 20 events for better context
        location_events = {}
        
        for event in recent_events:
            location = event.get('location', 'Unknown location')
            if location not in location_events:
                location_events[location] = []
            location_events[location].append(event)
        
        # Add location-specific context
        context += "Recent battlefield events by location:\n"
        for location, events in location_events.items():
            context += f"\n{location}:\n"
            for event in events:
                event_type = event.get('event_type', 'Unknown event')
                context += f"- {event_type}\n"
        
        context += f"\nUser question: {user_message}\n"
        context += "Please provide a detailed analysis based on the available data. For threat assessments, consider factors such as:\n"
        context += "1. Proximity to frontline activity\n"
        context += "2. Recent military movements\n"
        context += "3. Infrastructure importance\n"
        context += "4. Historical attack patterns\n"
        context += "5. Geographic significance"
        
        return context
    except Exception as e:
        logger.error(f"Error creating prompt: {str(e)}")
        raise

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        prompt = create_prompt(request.message)
        
        headers = {
            "Authorization": f"Bearer {LLAMA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a battlefield analyst AI assistant specializing in threat assessment and strategic analysis of the Russia-Ukraine conflict. Provide detailed, data-driven responses."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000
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
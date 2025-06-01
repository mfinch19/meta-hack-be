import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open('2024-05-24-2025-05-31-Russia-Ukraine.json') as f:
    events = json.load(f)

texts = []
metadata = []

for i, event in enumerate(events):
    text = f"{event.get('location', '')} ({event.get('admin1', '')}): {event.get('event_type', '')}, {event.get('sub_event_type', '')}. {event.get('notes', '')}"
    texts.append(text)
    metadata.append(event)

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index and metadata
faiss.write_index(index, "battlefield.index")
with open("battlefield_metadata.json", "w") as f:
    json.dump(metadata, f) 
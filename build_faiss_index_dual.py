import json
import faiss
import numpy as np
import ollama
import pandas as pd
from tqdm import tqdm
import datetime
import sys
import os
import argparse

print("Loading data...")

# Initialize Ollama client
ollama_client = ollama.Client()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Build FAISS index for battlefield events and NRC data.')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size for embedding generation')
args = parser.parse_args()

json_path = "2024-05-24-2025-05-31-Russia-Ukraine.json"
excel_path = "nrc_humanitarian_proximity_analysis_data_06_may_2025.xlsx"

def default_converter(o):
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    return str(o)

def get_embeddings(texts, batch_size=args.batch_size):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing embedding batches"):
        batch = texts[i:i + batch_size]
        print(f"Processing records: {i} - {min(i+batch_size, len(texts))}")
        response = ollama_client.embed(
            model="mxbai-embed-large:latest",
            input=batch
        )
        embeddings.extend(response['embeddings'])
    return np.array(embeddings)

with open(json_path) as f:
    events = json.load(f)
print(f"Loaded {len(events)} battlefield events.")

texts = []
metadata = []
for i, event in enumerate(tqdm(events, desc="Processing battlefield events")):
    text = (
        f"event_id_cnty: {event.get('event_id_cnty', '')}; "
        f"event_date: {event.get('event_date', '')}; "
        f"year: {event.get('year', '')}; "
        f"time_precision: {event.get('time_precision', '')}; "
        f"disorder_type: {event.get('disorder_type', '')}; "
        f"event_type: {event.get('event_type', '')}; "
        f"sub_event_type: {event.get('sub_event_type', '')}; "
        f"actor1: {event.get('actor1', '')}; "
        f"assoc_actor_1: {event.get('assoc_actor_1', '')}; "
        f"inter1: {event.get('inter1', '')}; "
        f"interaction: {event.get('interaction', '')}; "
        f"civilian_targeting: {event.get('civilian_targeting', '')}; "
        f"iso: {event.get('iso', '')}; "
        f"region: {event.get('region', '')}; "
        f"country: {event.get('country', '')}; "
        f"admin1: {event.get('admin1', '')}; "
        f"admin2: {event.get('admin2', '')}; "
        f"admin3: {event.get('admin3', '')}; "
        f"location: {event.get('location', '')}; "
        f"latitude: {event.get('latitude', '')}; "
        f"longitude: {event.get('longitude', '')}; "
        f"geo_precision: {event.get('geo_precision', '')}; "
        f"source: {event.get('source', '')}; "
        f"source_scale: {event.get('source_scale', '')}; "
        f"notes: {event.get('notes', '')}; "
        f"fatalities: {event.get('fatalities', '')}; "
        f"tags: {event.get('tags', '')}; "
        f"timestamp: {event.get('timestamp', '')}; "
        f"population_best: {event.get('population_best', '')}"
    )
    texts.append(text)
    metadata.append(event)

if os.path.exists(excel_path):
    # Print available sheets
    xl = pd.ExcelFile(excel_path)
    print("\nAvailable Excel sheets:", xl.sheet_names)
    
    # Load the Hromada Data sheet
    sheet_name = 'Hromada Data'
    print(f"\nLoading sheet: {sheet_name}")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    print(f"Loaded {len(df)} rows from NRC Humanitarian Proximity Analysis Excel.")
    print("\nColumns in Hromada Data sheet:")
    for col in df.columns:
        print(f"- {col}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing NRC Excel rows"):
        # Build a text string for embedding
        text = (
            f"Hromada PCode: {row.get('Hromada PCode', '')}; "
            f"Hromada Name (English): {row.get('Hromada Name (English)', '')}; "
            f"Hromada Name (Ukrainian): {row.get('Hromada Name (Ukrainian)', '')}; "
            f"Raion Name (English): {row.get('Raion Name (English)', '')}; "
            f"Raion Name (Ukrainian): {row.get('Raion Name (Ukrainian)', '')}; "
            f"Raion PCode: {row.get('Raion PCode', '')}; "
            f"Oblast Name (English): {row.get('Oblast Name (English)', '')}; "
            f"Oblast Name (Ukrainian): {row.get('Oblast Name (Ukrainian)', '')}; "
            f"Oblast PCode: {row.get('Oblast PCode', '')}; "
            f"urban_percentage: {row.get('urban_percentage', '')}; "
            f"rural_percentage: {row.get('rural_percentage', '')}; "
            f"intersection_percentage_5km: {row.get('intersection_percentage_5km', '')}; "
            f"intersection_percentage_15km: {row.get('intersection_percentage_15km', '')}; "
            f"intersection_percentage_20km: {row.get('intersection_percentage_20km', '')}; "
            f"intersection_percentage_30km: {row.get('intersection_percentage_30km', '')}; "
            f"intersection_percentage_50km: {row.get('intersection_percentage_50km', '')}; "
            f"incidents_1_Month: {row.get('incidents_1_Month', '')}; "
            f"incidents_3_Months: {row.get('incidents_3_Months', '')}; "
            f"incidents_6_Months: {row.get('incidents_6_Months', '')}; "
            f"incidents_9_Months: {row.get('incidents_9_Months', '')}; "
            f"incidents_12_Month: {row.get('incidents_12_Month', '')}"
        )
        # Add to texts and metadata
        texts.append(text)
        metadata.append({k: default_converter(v) for k, v in row.to_dict().items()})
else:
    print("NRC Humanitarian Proximity Analysis Excel file not found. Skipping integration.")

print(f"\nTotal texts to embed: {len(texts)}")
print("Generating embeddings...")
try:
    embeddings = get_embeddings(texts)
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("Saving index and metadata...")
    faiss.write_index(index, "battlefield.index")
    with open("battlefield_metadata.json", "w") as f:
        json.dump(metadata, f, default=default_converter)
    print("Done!")
except KeyboardInterrupt:
    print("\nEmbedding generation interrupted by user. Saving partial results...")
    if 'embeddings' in locals():
        print(f"Saving {len(embeddings)} embeddings...")
        faiss.write_index(index, "battlefield.index")
        with open("battlefield_metadata.json", "w") as f:
            json.dump(metadata[:len(embeddings)], f, default=default_converter)
    sys.exit(1)
except Exception as e:
    print(f"\nError during embedding generation: {str(e)}")
    sys.exit(1)

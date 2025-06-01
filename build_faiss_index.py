import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from fastkml import kml
import zipfile
import os

print("Loading data...")
model = SentenceTransformer('all-MiniLM-L6-v2')

with open('2024-05-24-2025-05-31-Russia-Ukraine.json') as f:
    events = json.load(f)
print(f"Loaded {len(events)} battlefield events.")

# --- Project Owl OSINT Integration ---
osint_events = []
kmz_path = "Ukraine Control Map v2.kmz"
extract_dir = "kmz_extracted"
if os.path.exists(kmz_path):
    with zipfile.ZipFile(kmz_path, 'r') as z:
        z.extractall(extract_dir)
    kml_file = os.path.join(extract_dir, "doc.kml")
    if os.path.exists(kml_file):
        with open(kml_file, 'rt', encoding='utf-8') as f:
            doc = f.read()
        k = kml.KML()
        k.from_string(doc.encode('utf-8'))
        for feature in k.features():
            for subfeature in feature.features():
                for placemark in subfeature.features():
                    osint_event = {
                        'event_id_cnty': 'osint_' + (placemark.name or ''),
                        'event_date': '',  # Optionally parse from description
                        'year': '',
                        'time_precision': '',
                        'disorder_type': '',
                        'event_type': 'OSINT',
                        'sub_event_type': '',
                        'actor1': '',
                        'assoc_actor_1': '',
                        'inter1': '',
                        'interaction': '',
                        'civilian_targeting': '',
                        'iso': '',
                        'region': '',
                        'country': 'Ukraine',
                        'admin1': '',
                        'admin2': '',
                        'admin3': '',
                        'location': placemark.name or '',
                        'latitude': placemark.geometry.y if placemark.geometry else '',
                        'longitude': placemark.geometry.x if placemark.geometry else '',
                        'geo_precision': '',
                        'source': 'Project Owl OSINT',
                        'source_scale': '',
                        'notes': placemark.description or '',
                        'fatalities': '',
                        'tags': '',
                        'timestamp': '',
                        'population_best': ''
                    }
                    osint_events.append(osint_event)
        print(f"Loaded {len(osint_events)} OSINT events from Project Owl.")
    else:
        print("KML file not found in extracted KMZ.")
else:
    print("Project Owl KMZ file not found. Skipping OSINT integration.")

# Merge OSINT with main events
if osint_events:
    events.extend(osint_events)
    print(f"Total events after OSINT integration: {len(events)}")

texts = []
metadata = []

for i, event in enumerate(events):
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

print("Generating embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("Saving index and metadata...")
faiss.write_index(index, "battlefield.index")
with open("battlefield_metadata.json", "w") as f:
    json.dump(metadata, f)
print("Done!") 
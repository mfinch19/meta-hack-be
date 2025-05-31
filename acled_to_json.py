import csv
import json

csv_file_path = '/Users/fabdelmoneum/Desktop/LLama Hackathon/2024-05-24-2025-05-31-Russia-Ukraine.csv'
json_file_path = '/Users/fabdelmoneum/Desktop/LLama Hackathon/2024-05-24-2025-05-31-Russia-Ukraine.json'

# Adjust delimiter and quotechar if needed
with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=';', quotechar='"')
    # Read the header
    headers = next(reader)
    # Remove empty headers if any (common with trailing delimiters)
    headers = [h for h in headers if h.strip() != '']

    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json_file.write('[\n')
        first = True
        for row in reader:
            # Remove trailing empty fields if any
            row = row[:len(headers)]
            item = dict(zip(headers, row))
            if not first:
                json_file.write(',\n')
            json.dump(item, json_file, ensure_ascii=False)
            first = False
        json_file.write('\n]')
print(f"Done! JSON saved to {json_file_path}")
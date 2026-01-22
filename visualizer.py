import re
from collections import defaultdict

def parse_pipeline_log(file_path):
    with open(file_path, 'r') as f:
        log_content = f.read()

    # Split into individual documents
    docs = log_content.split("DOCUMENT_START |")
    visualizer_data = []

    for doc in docs[1:]:  # Skip the batch header
        doc_id = re.search(r"DocID: ([\w_]+)", doc).group(1)
        timestamp = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", doc).group(1)
        
        # Track engine outputs per field
        field_results = defaultdict(dict)
        
        # Find all FINAL_RESULT entries
        final_results = re.findall(
            r"FINAL_RESULT \| Field: (\w+) \| Value: (.*?) \| Conf: ([\d.]+) \| Source: ([\w_]+)", 
            doc
        )
        
        for field, value, conf, source in final_results:
            field_results[field] = {
                "value": value,
                "confidence": conf,
                "winning_engine": source
            }

        visualizer_data.append({
            "timestamp": timestamp,
            "doc_id": doc_id,
            "fields": field_results
        })

    return visualizer_data

def display_visualizer(data):
    print(f"{'TIMESTAMP':<20} | {'DOC ID':<30} | {'FIELD':<15} | {'WINNING ENGINE'}")
    print("-" * 90)
    for entry in data:
        for field, details in entry['fields'].items():
            print(f"{entry['timestamp']:<20} | {entry['doc_id'][:30]:<30} | {field:<15} | {details['winning_engine']}")

# Usage
data = parse_pipeline_log('pipeline_run_20260122_180518.log')
display_visualizer(data)
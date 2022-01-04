import argparse
import jsonlines
import json
import numpy as np
from pathlib import Path
def jsonl_to_json(file_path, output_path):
    l = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            l.append(obj)
    with open(output_path, 'w') as fp:
        json.dump({'data': l}, fp, indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default = Path('./data'))
    parser.add_argument("--json_data_dir", type=Path, default = Path('./data_json'))
    args = parser.parse_args()
    return args
            
if __name__ == "__main__":
    data_dir = Path('./data')
    json_data_dir = Path('./data_json')
    if not json_data_dir.exists():
        json_data_dir.mkdir()
    file_name = ['train', 'public']
    for name in file_name:
        jsonl_to_json(data_dir / f"{name}.jsonl", json_data_dir / f"{name}.json")
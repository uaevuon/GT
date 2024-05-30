import os
import json
import shutil

tag = '아라벨'

# Function to extract tags from a JSON file
def extract_tags(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('tags', [])

# Function to traverse directories and move folders with specific tags
def traverse_directories(root_dir):
    raw_dir = os.path.join(root_dir, 'raw')
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(root, file)
                tags = extract_tags(json_file)
                if tag in tags:
                    print("Processing file:", file)
                    folder_path = os.path.dirname(json_file)
                    destination_folder = os.path.join(root_dir, 'pixiv')
                    shutil.move(folder_path, destination_folder)


def main():
    root_dir = r'E:\code\capstone\data'  
    
    traverse_directories(root_dir)

if __name__ == "__main__":
    main()
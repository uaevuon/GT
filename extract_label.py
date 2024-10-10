import os
import urllib
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, json_file, transform=None):
        self.data_dir = data_dir
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.transform = transform

        # Create a set to store all unique labels
        all_labels = set()
        for item in self.data:
            if 'choice' in item:
                if isinstance(item['choice'], dict):
                    all_labels.update(item['choice']['choices'])
                else:
                    all_labels.add(item['choice'])

        # Create a dictionary to map labels to indices
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path_encoded = item['image'].split('?d=')[-1]
        image_path = os.path.join(self.data_dir, urllib.parse.unquote(image_path_encoded))
        #print(f"Image path: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        # Create a tensor of zeros with a length equal to the total number of classes
        num_classes = len(self.label_to_idx)
        labels = torch.zeros(num_classes, dtype=torch.float32)

        
        # Set the corresponding elements in the tensor to 1 for the labels present
        if 'choice' in item:
            if isinstance(item['choice'], dict):
                choices = item['choice']['choices']
            else:
                choices = [item['choice']]
            for choice in choices:
                label_idx = self.label_to_idx[choice]
                labels[label_idx] = 1



                
        return image, labels
data_dir = r'E:\code\capstone\data'
json_file = r'E:\code\capstone\GT\label-studio-export\project-1-at-2024-05-28-01-40-d3301fc8.json'    


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset(data_dir, json_file, transform=transform)    
# Assuming 'dataset' is your CustomDataset instance
label_map = {v: k for k, v in dataset.label_to_idx.items()}

with open('label_map.json', 'w') as f:
    json.dump(label_map, f)
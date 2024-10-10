import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
from PIL import Image

# Define the model
class MultiLabelMobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelMobileNetV3, self).__init__()
        self.base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
    
# Function to load the model
def load_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiLabelMobileNetV3(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model, device

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Inference function
def classify_image(model, image_path, device):
    model.eval()
    image_tensor = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output)
        
        # Get the predicted label (you can adjust this part to suit your needs)
        predicted_label = torch.argmax(probabilities, dim=1).item()
    
    return predicted_label
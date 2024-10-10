import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from torchvision.models import mobilenet_v3_small

class MultiLabelMobileNetV3(torch.nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelMobileNetV3, self).__init__()
        self.base_model = mobilenet_v3_small(weights=None)
        self.base_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(576, 1024),
            torch.nn.Hardswish(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)

class GuardianTalesPredictor:
    def __init__(self, model_path, label_map_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct absolute paths
        model_path = os.path.join(current_dir, model_path)
        label_map_path = os.path.join(current_dir, label_map_path)

        # Load label map
        with open(label_map_path, 'r') as f:
            self.idx_to_label = json.load(f)
        self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}  # Ensure keys are integers
        
        # Load model
        num_classes = len(self.idx_to_label)
        self.model = MultiLabelMobileNetV3(num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.sigmoid(output)
            predicted_idx = torch.argmax(probabilities).item()
            predicted_label = self.idx_to_label[predicted_idx]
            confidence = probabilities[0][predicted_idx].item()

        return predicted_label, confidence

# Initialize the predictor
predictor = GuardianTalesPredictor('pth/mobilenetv3-nofreeze-105-loss.pth', 'label_map.json')

def predict_image(image_path):
    """
    Predict the label of a Guardian Tales character image.
    
    Args:
    image_path (str): Path to the image file.
    
    Returns:
    tuple: (predicted_label, confidence)
    """
    return predictor.predict_image(image_path)

if __name__ == "__main__":
    # Example usage
    test_image_path = r"C:\Users\yaong\Desktop\1716038013.jpeg"
    label, confidence = predict_image(test_image_path)
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.4f}")

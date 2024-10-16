import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Assuming the model definition is already available, import your trained model class
from main import pil_loader,RichIQA

# Define the path to the saved model
MODEL_PATH = 'pretrined_Model.pkl'  # Update this path to the location of your trained model

# Load the trained model


def load_model(model_path):
    
    model = torch.nn.DataParallel(RichIQA(options=None), device_ids=[0]).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Define preprocessing transformations (resize, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the required input size for the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict image quality
def predict_image_quality(model, image_path):
    # Load and preprocess the image
    image = pil_loader(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(image)
    
    # Assuming the output is a quality score, convert it to a readable format
    quality_score = output.item()  # Modify if output is not a single value
    return quality_score

if __name__ == "__main__":
    # Load the model
    model = load_model(MODEL_PATH)

    # Define the image path (you can change this to any image you want to predict)
    image_path = 'test.jpg'  # Update this path to your input image

    # Predict the image quality
    quality_score = predict_image_quality(model, image_path)

    # Print the result
    print(f'Predicted Image Quality Score: {quality_score}')

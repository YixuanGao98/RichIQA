import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
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
    quality_score = output.cpu().detach().numpy()  # Modify if output is not a single value
    si = np.arange(1, 6, 1)#
    # si = np.arange(5, 105, 10)#
    mean = np.sum(quality_score * si)
    return quality_score,mean

if __name__ == "__main__":
    # Load the model
    model = load_model(MODEL_PATH)

    # Define the image path (you can change this to any image you want to predict)
    image_path = 'test.jpg'  # Update this path to your input image

    # Predict the image quality
    quality_score,mean = predict_image_quality(model, image_path)

    # Print the result
    print(f'Predicted Image Quality Score  Distribution: {quality_score}')
    print(f'Predicted mos: {mean}')


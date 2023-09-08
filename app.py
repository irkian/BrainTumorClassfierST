import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Define the Cifar10CnnModel class first
class Cifar10CnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256*8*8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)

# Load the PyTorch model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('model.pth', map_location=device)

model.eval()

# Define transformations for image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Define a function to make predictions
def predict_image_class(image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = output.max(1)
    return predicted_class.item()

# Streamlit UI
st.title('Brain Tumor Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Classify'):
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        predicted_class = predict_image_class(image)
        st.write(f'Predicted Class: {class_names[predicted_class]}')

# Create an instance of the Cifar10CnnModel after its definition
model_instance = Cifar10CnnModel()
model_instance = to_device(model_instance, device)

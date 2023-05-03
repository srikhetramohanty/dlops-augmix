import torch
import streamlit as st
from PIL import Image
from torchvision import models, transforms

#Load the model
model = torch.load('model_best.pth', map_location=torch.device('cpu'))
model.eval()

# load the pre-trained model
# model = models.resnet18(pretrained=True)
# model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the prediction function
def predict(image):
    # img = transform(image).unsqueeze(0)
    # output = model(img)
    # _, preds = torch.max(output, 1)
    
    img_tensor = transform(image)

    # add batch dimension to the image
    img_tensor = img_tensor.unsqueeze(0)

    # pass the image through the model
    with torch.no_grad():
        output = model(img_tensor)

    # get the predicted class label
    _, predicted = torch.max(output.data, 1)
    label = predicted.item()    
    
    return label

# Create the Streamlit app
def main():
    st.title('Image Classification through Augmix learning - \n Srikhetra Mohanty (M21AIE260) & Rakesh Sahoo (M21AIE246)')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        pred = predict(image)
        st.write(f'Prediction: {pred}')

if __name__ == '__main__':
    main()

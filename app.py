import torch
import streamlit as st
from PIL import Image
from torchvision import models, transforms
from allconv import *

#Load the model
# model = torch.load('model_best.pth.tar', map_location=torch.device('cpu'))
# model.load_state_dict(torch.load('model_best.pth.tar'))
#model.eval()

###########################################################################################################
###########################################################################################################
model = AllConvNet(num_classes=100)
#model.load_state_dict(torch.load('model_best.pth.tar', map_location=torch.device('cpu')),strict=False)
checkpoint = torch.load('model_best_v2.pth.tar', map_location=torch.device('cpu'))
dict_ = {}

for k,v in checkpoint["state_dict"].items():
    dict_.update({k.replace("module.",""):v})

model.load_state_dict(dict_)    

###########################################################################################################
###########################################################################################################

fine_labels = [
    'apple',  # id 0
    'aquarium_fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'computer_keyboard',
    'lamp',
    'lawn_mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple_tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak_tree',
    'orange',
    'orchid',
    'otter',
    'palm_tree',
    'pear',
    'pickup_truck',
    'pine_tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet_pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow_tree',
    'wolf',
    'woman',
    'worm']

label_map = {}

#Mapping fine labels to index
for id_, label in enumerate(fine_labels):
    label_map.update({id_:label})

#print(model)

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
        op_string = "The prediction is : " + str(pred)+ " -> " + str(label_map[pred])
        st.write(f'Prediction: {op_string}')

if __name__ == '__main__':
    main()

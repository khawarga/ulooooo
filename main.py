import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

class_names = ['Non Venomous', 'Venomous']

# load model
model_ft = models.vgg16(weights=True)
num_ftrs = model_ft.classifier[0].out_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft.load_state_dict(torch.load('model.pth'))

# image transfromer
converter = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

st.set_page_config(layout="wide", page_title="Venomous or Non Venomous Snake Prediction")    

hide_default_format = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title('Venomous or Non Venomous Snake Prediction')
st.write('This prediction is 80% accurate')

file = st.file_uploader("Insert snake picture", type=["jpg", "jpeg", "png"])

if file:
    # Hide filename on UI
    st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
        unsafe_allow_html=True)

if file is not None:
    image = Image.open(file)

    tensor_img = converter(image)

    # predictions
    model_ft.eval()
    output = model_ft(tensor_img)
    _, preds = torch.max(output, 1)
    st.write(f'Predicted: {class_names[preds]}')
    st.image(image)
    if(class_names[preds] == "Non Venomous"):
        st.write(f'Characteristic of non venomous snakes')
        st.write(f'1. Non Venomous snakes have rounder head')
        st.write(f'2. Non Venomous snakes have rounder pupils that fill all of its eyes')
        st.write(f'3. Non Venomous snakes dont have rattle on its tail ')
        st.write(f'4. Non Venomous snakes have solid color and dont have any motif on its body')

    
    if(class_names[preds] == "Venomous"):
        st.write(f'Characteristic of venomous snakes')
        st.write(f'1. Venomous snakes have triangle shaped head')
        st.write(f'2. Venomous snakes have vertical pupils that looks menacing')
        st.write(f'3. Some Venomous snakes have rattle on its tail ')
        st.write(f'4. Venomous snakes have bright color and have interesting motif on its body')

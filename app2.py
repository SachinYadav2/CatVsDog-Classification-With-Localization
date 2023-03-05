import pickle
import streamlit as st
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
model = load_model('model.h5')
# Your Page Title
st.title("Cat Vs Dog Classifier")

# UPload Image 
uploaded_image = st.file_uploader("Chose an Image")



def save_uploaded_image(uploaded_image):

    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())

        return (uploaded_image.name)
    except:
        return False

x = save_uploaded_image(uploaded_image)



if uploaded_image is not None:
    col1,col2 = st.columns(2)
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)

        with col1:
            st.header('Your uploaded image')
            
            m=st.image(display_image,width=300)
   
        def extract_features(img_path,model):
                


            pred_img = cv2.imread(img_path)
            pred_img = cv2.resize(pred_img,(224, 224))


            c,bnd = model.predict(np.asarray([pred_img]))

            bnd = np.round(bnd)
            bnd = np.array(bnd.astype('int64'))
            xmin , ymin , xmax , ymax      =    bnd[0][0] , bnd[0][1] ,bnd[0][2] ,bnd[0][3] 

            img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
            bnd_img=cv2.rectangle(img ,(xmin,ymin),(xmax,ymax),(255,0,0),2) 
            with col2:
                st.header('Your Output image')
            
                m=st.image(bnd_img,width=300)



              





        uploaded = 'uploads'
        x = str(x)
        img_path = os.path.join(uploaded,x)
        img_path = str(img_path)
        extract_features(img_path,model)




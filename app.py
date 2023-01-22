import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import scipy
from scipy import stats
st.set_option('deprecation.showfileUploaderEncoding', False)
# Loading saved model from Drive.
from keras.models import load_model
model = load_model('/content/drive/My Drive/Deep Learning FDP/Face dataset/Facemodel.h5')
FACE_CLASSES = ['ben_afflek', 'elton_john','jerry_seinfeld','madonna','mindy_kaling']
html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Vepsun Technologies</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Session on Transfer Learning</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Celebrity Face Recognition
         """
         )
file= st.file_uploader("Please upload image of ['ben_afflek', 'elton_john','jerry_seinfeld','madonna','mindy_kaling']", type=("jpg", "png"))

import cv2
from  PIL import Image, ImageOps
def import_and_predict(image_data):
  #x = cv2.resize(image_data, (48, 48)) 
  #img = image.load_img(image_data, target_size=(48, 48))
  #x = image.img_to_array(img)
  size=(224, 224)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  features = model.predict(img_reshape)
  print(features)
  label_index=features.argmax()
  print(label_index)
  print("Model prediction :", FACE_CLASSES[label_index])
  
  return FACE_CLASSES[label_index]
if file is None:
  st.text("Please upload an Image file")
else:
  image=Image.open(file)
  #image=np.array(image)
  #file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  #image = cv2.imdecode(file_bytes, 1)
  st.image(image,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Predict Expression"):
  result=import_and_predict(image)
  st.success('Model has predicted the image expression is   {}'.format(result))
if st.button("About"):
  st.header(" Deepak Moud")
  st.subheader("Neural Network Trainer")
  
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Deep Learning Transfer learning</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)

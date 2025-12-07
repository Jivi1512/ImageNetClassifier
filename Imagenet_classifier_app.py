import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model=ResNet50(weights='imagenet')

st.title("Capture an image!!")
img=st.camera_input("Take a picture!!")

if img is not None:
    pil_img = Image.open(img).convert("RGB")
    img_resize=pil_img.resize((224,224))
    img_arr=image.img_to_array(img_resize)
    img_arr=np.expand_dims(img_arr, axis=0)
    img_arr=preprocess_input(img_arr) 
    with st.spinner("Processing..."):
        preds = model.predict(img_arr)
        output_list=decode_predictions(preds, top=3)[0]
    st.success("Top 3 Predictions:")
    for i in output_list:
        st.write(f"{i[1]}-\tConfidence level:{i[2]*100:.2f} %")
    plt.imshow(pil_img)
    plt.title(f"{output_list[0][1]} {output_list[0][2]*100:.2f} %")
    plt.axis("off")
    st.pyplot(plt)

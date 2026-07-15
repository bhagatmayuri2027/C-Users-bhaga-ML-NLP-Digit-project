import pandas as pd
import numpy as np
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model


model=load_model("digit_recognition_model.keras")
st.title("Handwritten Digit Recognition")

canvas_result=st_canvas(
    fill_color="#00000000",
    stroke_color="#FFFFFFFF",
    stroke_width=10,
    background_color="#000000",
    width=400,
    height=400,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Predict"):
    st.write("Predicting...........")

    img=canvas_result.image_data.astype(np.uint8)
    #grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #grey_img=img[:,:, 3]
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)    
    grey_img=cv2.resize(grey_img,(28,28))

    grey_img=grey_img/255.0

    grey_img=grey_img.reshape(-1,784)

    result=model.predict(grey_img)

    index=np.argmax(result)

    st.write(f"The predicted digit is:{index}")




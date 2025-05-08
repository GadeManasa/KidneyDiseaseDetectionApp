import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Set up the Streamlit app layout and widgets
st.set_page_config(page_title="Kidney Disease Classification App", page_icon=":kidney:", layout="wide")

st.title("Kidney Disease Classification App")

st.markdown("""
This application classifies kidney diseases based on CT scan images.
Simply upload your kidney CT scan image and the app will predict whether it's normal or shows signs of cyst, stone, or tumor.
""")

st.sidebar.header("INFO")
st.sidebar.markdown("""
This application uses a convolutional neural network (CNN) model trained on a dataset of kidney CT scan images.
It can classify images into four categories: Normal, Cyst, Stone, and Tumor.
""")

# Online background image
@st.cache_resource()
def load_bg_image():
    response = requests.get("https://wallpaper-mania.com/wp-content/uploads/2018/09/High_resolution_wallpaper_background_ID_77701450911.jpg")
    img = Image.open(BytesIO(response.content))
    return img

bg_img = load_bg_image()
st.sidebar.image(bg_img, use_column_width=True)

# Informative sections
expander_kidney_info = st.sidebar.expander("Kidney Disease Information")
with expander_kidney_info:
    st.markdown("""
    Kidney diseases are conditions that affect the normal functioning of kidneys. They can be caused by various factors, such as infections, autoimmune disorders, and genetic conditions.
    Some common kidney diseases include kidney stones, cysts, and tumors. Early detection and treatment are essential to prevent the progression of the disease and protect kidney function.
    """)

expander_prevention = st.sidebar.expander("Prevention Tips")
with expander_prevention:
    st.markdown("""
    - Maintain a healthy diet: Consume a balanced diet with low sodium, high-fiber foods, and avoid processed foods.
    - Stay hydrated: Drink enough water to help flush out toxins and prevent kidney stones.
    - Exercise regularly: Engage in physical activity to maintain a healthy weight and blood pressure.
    - Avoid smoking and excessive alcohol consumption: These habits can damage kidney function.
    - Monitor blood pressure and blood sugar levels: High blood pressure and diabetes can lead to kidney damage if not controlled.
    """)

expander_doctor = st.sidebar.expander("When to See a Doctor")
with expander_doctor:
    st.markdown("""
    If you experience symptoms such as persistent pain in the lower back, side or abdomen, blood in the urine, frequent urination, or a decrease in urine output, consult a healthcare professional for evaluation and advice.
    """)

# Function to load the trained model
# @st.cache_data()
def load_model():
    model = tf.keras.models.load_model("kid_desease_classification_model_CNN.h5")
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((200, 200), Image.LANCZOS)
    img = img.convert('L')
    img_arr = np.asarray(img)
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape((1,200, 200, 1))
    return img_arr

#Function to make a prediction using the loaded model
def make_prediction(model, img_arr):
    prediction = model.predict(img_arr)
    return prediction

# User interaction logic for uploading an image, loading the trained model, making predictions, and displaying the result
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
    img_arr = preprocess_image(uploaded_file)
    model = load_model()
    prediction = make_prediction(model, img_arr)
    
    # Display the result
    categories = ['Cyst', 'Normal', 'Stone', 'Tumor']
    pred_index = np.argmax(prediction)
    st.write(f"Prediction: {categories[pred_index]}")
    st.write(f"Probability: {round(prediction[0][pred_index]*100, 2)}%")


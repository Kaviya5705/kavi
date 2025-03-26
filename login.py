

import streamlit as st
import base64

# Set up the page
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")

# Function to encode the image into Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"  # Change 'png' to 'jpg' if needed

# Function to set the background image
def set_background(image_path):
    base64_image = get_base64_image(image_path)
    background_style = f"""
    <style>
        .stApp {{
            background-image: url("{base64_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Set the background image (Change the path to your actual image file)
set_background("C:/project/work/image.jpg")  

# Title and Introduction
st.title("Welcome to Diabetic Retinopathy Detection System")
st.markdown("""
### A Simple Web App for Awareness and Assistance  
This platform aims to educate and assist in understanding *Diabetic Retinopathy*, a complication of diabetes that affects the eyes.  
""")

# About Section
st.header("About Diabetic Retinopathy")
st.write("""
Diabetic Retinopathy is a diabetes-related eye disease that can lead to vision loss if not detected early.  
It occurs when high blood sugar levels cause damage to the blood vessels in the retina.  
Symptoms may include:
- Blurred vision
- Dark spots in vision
- Vision loss in severe cases

Early detection through regular eye check-ups is crucial for preventing complications.
""")

# Call-to-Action
st.header("How This App Helps")
st.write("""
- Provides information on *Diabetic Retinopathy*  
- Encourages *early detection and medical consultation*  
- Future updates may include AI-powered analysis of retinal images  
""")

# Navigation Buttons
if st.button("Admin Page"):
    import subprocess
    subprocess.run(['streamlit', 'run', 'admin.py'])

if st.button("User Page"):
    import subprocess
    subprocess.run(['streamlit', 'run', 'register.py'])

# Footer
st.markdown("""
---
*Disclaimer:* This application is for informational purposes only. Please consult an ophthalmologist for a medical diagnosis.
""")

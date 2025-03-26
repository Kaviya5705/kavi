
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:46:41 2025

@author: ARUNKUMAR
"""

import streamlit as st
import os
import base64



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


# Streamlit interface for the user page
st.title('Diabetic Retinopathy Detection - User Results')

# Step 1: Request user input (user's name)
user_input = st.text_input("Enter your name")

# If the user provides a name, proceed to fetch the result
if user_input:
    result_directory = user_input  # Assuming the results are stored under the user's name
    result_file = os.path.join(result_directory+"result.txt")

    # Check if the result file exists
    if os.path.exists(result_file):
        # Read the result from the file****
        with open(result_file, "r") as file:
            result_data = file.read()

        # Display the results to the user
        st.write("### Prediction Result:")
        st.text(result_data)
        
        # Optionally, provide the ability to download the results
        st.download_button(label="Download Results", data=result_data, file_name="result.txt", mime="text/plain")
    else:
        # Provide debug message if the file is not found
        st.write("No results found for the user.")
        st.write(f"Checked path: {result_file}")
        st.write("Please ensure that the admin has uploaded the results.")
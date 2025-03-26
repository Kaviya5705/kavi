# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:49:37 2025

@author: ARUNKUMAR
"""

import streamlit as st
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


# Initialize session state for login check
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Admin login function
def authenticate_admin(username, password):
    # Hardcoded admin credentials (In practice, use a more secure method)
    if username == "admin" and password == "admin123":
        st.session_state.logged_in = True
        return True
    else:
        st.session_state.logged_in = False
        return False

# Show login page if not logged in
if not st.session_state.logged_in:
    st.title("Admin Login Page")

    # Create username and password fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    # Check login button
    if st.button("Login"):
        if authenticate_admin(username, password):
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            # Rerun to refresh and show next steps
            import subprocess
            subprocess.run(['streamlit','run','app.py'])
        else:
            st.error("Invalid credentials. Please try again.")
else:
    # Once logged in, show a success message
    st.success("Welcome Admin! You are now logged in.")
    # You can add further content here to navigate to another page or show content.

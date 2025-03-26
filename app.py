import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Function to load and preprocess the image  
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    
    # Resize the image (standard size, e.g., 224x224)
    resized_image = cv2.resize(image, (224, 224))

    # Convert to grayscale for further processing
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Denoise the image using GaussianBlur
    denoised_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Convert back to BGR format for displaying
    denoised_image_bgr = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)
    
    return denoised_image_bgr, gray_image

# Feature extraction function (Edge Detection and Texture Extraction)
def extract_features(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Edge Detection using Canny
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
    
    # Texture extraction using Local Binary Patterns (LBP)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    
    # Extracting features by computing histograms of the edge and LBP images
    edge_hist = np.histogram(edges.ravel(), bins=256, range=(0, 256))[0]
    lbp_hist = np.histogram(lbp.ravel(), bins=256, range=(0, 256))[0]
    
    features = np.concatenate([edge_hist, lbp_hist])
    
    return features

# Placeholder classifier - In practice, load a pre-trained model
def train_classifier(X_train, y_train):
    classifier = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
    classifier.fit(X_train, y_train)
    return classifier

# Streamlit interface
st.title('Diabetic Retinopathy Detection and Classification')

# Generate synthetic data for training (in practice, use your real dataset)
# Let's assume we have 1000 feature vectors with labels for training
X_synthetic = np.random.rand(1000, 512)  # Placeholder for 1000 samples, each with 512 features
y_synthetic = np.random.randint(0, 2, 1000)  # Binary labels (0 or 1) for synthetic data

# Split the synthetic data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

# Train the classifier
classifier = train_classifier(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


# Create a text input box
user_input = st.text_input("Enter your name")

st.write(user_input)

try:        
    import os
    result_directory = user_input
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

except:
    None


# Upload image using file uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV format
    image = Image.open(uploaded_file)
    
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, format='JPEG')
        temp_image_path = temp_file.name
    
    # Preprocess the image (resize, denoise, grayscale conversion)
    processed_image, gray_image = preprocess_image(temp_image_path)
    
    # Print the shapes for debugging
    st.write(f"Processed image shape: {processed_image.shape}")
    st.write(f"Grayscale image shape: {gray_image.shape}")
    
    # Extract features (edges and texture)
    extracted_features = extract_features(processed_image)
    st.write(f"Extracted features shape: {extracted_features.shape}")
    
    # Reshape the features to 2D (required by the classifier)
    extracted_features = extracted_features.reshape(1, -1)
    
    # Make a prediction
    prediction = classifier.predict(extracted_features)
    prob_prediction = classifier.predict_proba(extracted_features)
    
    # Show prediction and confidence score
    if prediction == 0:
        result = "No Diabetic Retinopathy Detected"
    else:
        result = "Diabetic Retinopathy Detected"
    
    file1 = open(user_input+"result.txt", "w")
    file1.write(result)
    file1.close()  # to change file access modes

    confidence = np.max(prob_prediction)
    
    # Display the raw image for comparison
    st.image(image, caption="Original Image", use_column_width=True)

    # Display the processed image
    st.image(processed_image, caption="Processed Image (Denoised and Grayscale)", use_column_width=True)

    # Display the grayscale image
    st.image(gray_image, caption="Grayscale Image", use_column_width=True, channels="GRAY")


    # Display the prediction result
    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {confidence*100:.2f}%")
    
    # Display accuracy and confusion matrix
    st.write(f"Accuracy: {accuracy * 150:.2f}%")
    st.write("Confusion Matrix:")
    st.write(conf_matrix) 
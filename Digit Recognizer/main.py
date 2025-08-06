import streamlit as st
import numpy as np 
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Streamlit page config
st.set_page_config(page_title="MSINT Digit Recognizer", page_icon="🔢", layout="centered")
st.title("MSINT DIGIT RECOGNIZER")
st.write("Upload an image of a handwritten digit (0-9) to predict the number using a K-Nearest Neighbors model.")

# Load and train model (cached)
@st.cache_resource
def load_and_train_data():
    df = pd.read_csv("train.csv")  # MNIST dataset
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.96)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    knn = KNeighborsClassifier()
    knn.fit(X_train_pca, Y_train)

    return scaler, pca, knn

scaler, pca, knn = load_and_train_data()

# 🛠️ Fix: Don't normalize to 0-1 before scaling!
def preprocess_image(image):
    img = image.convert('L')  # Grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)

    # Auto-invert if background is white
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Show preprocessed image
    st.image(img_array, caption="Preprocessed Image", width=150, clamp=True)

    # Flatten to 1D
    img_flat = img_array.flatten().reshape(1, -1)  # Shape (1, 784)

    # Scale and transform (same as training)
    img_scaled = scaler.transform(img_flat)
    img_pca = pca.transform(img_scaled)

    return img_pca

# Image uploader
uploaded_file = st.file_uploader("Choose an image of a handwritten digit", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Use session state to avoid stale predictions
    if "last_image_bytes" not in st.session_state or st.session_state["last_image_bytes"] != uploaded_file.getvalue():
        st.session_state["last_image_bytes"] = uploaded_file.getvalue()

        # Preprocess and predict
        img_pca = preprocess_image(image)

        try:
            prediction = knn.predict(img_pca)
            probability = np.max(knn.predict_proba(img_pca))
        except Exception as e:
            st.error(f"Prediction error: {e}")
            prediction, probability = None, None

        st.session_state["last_prediction"] = (prediction[0], probability) if prediction is not None else (None, None)

    # Show prediction
    pred_digit, prob = st.session_state.get("last_prediction", (None, None))
    if pred_digit is not None:
        st.subheader("Prediction")
        st.write(f"The model predicts the digit is **{pred_digit}** with confidence **{prob:.2%}**")
    else:
        st.warning("Could not predict from the image.")

    st.info("Tip: Upload a clear image with **white digit on black background**, centered and not noisy.")

st.markdown("---")
st.write("Built with Streamlit | Model accuracy: ~96% | Trained on MNIST dataset")

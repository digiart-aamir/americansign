import streamlit as st
import tensorflow as tf
from tensorflow.keras.saving import load_model
import numpy as np
from PIL import Image

# Set the title and icon
st.set_page_config(
    page_title="American Sign Prediction using CNN model",
    page_icon=":bar_chart:"  # data analysis emoji
)

# Display the app title
st.title("American Sign Prediction using CNN model")

# Create two columns for layout
col1, col2 = st.columns(2)

# Column 1: Sample Image
with col1:
    st.header("Sample Image")
    sample_image = Image.open('sample.png')  # Load the sample image
    st.image(sample_image, caption="Sample Image", use_column_width=True)

# Column 2: Upload and Prediction
with col2:
    # Load the model
    def load_cnn_model():
        model = load_model('cnn_model.keras')
        return model

    # Load the model and display success message
    model = load_cnn_model()

    # Upload image section
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Normalize the image
        image_np = image_np / 255.0

        # Reshape the image to fit the model input shape (1, 28, 28, 1)
        image_np = image_np.reshape(-1, 28, 28, 1)

        # Display the processed image
        st.image(image, caption="Processed Image", width=150)

        # Mapping for actual labels (A-I, K-Y, skipping J)
        label_map = {i: chr(65 + i) if i < 9 else chr(65 + i + 1) for i in range(24)}  # Map 0-23 to 'A'-'I' and 'K'-'Y'

        # Model prediction
        if st.button("Predict"):
            prediction = model.predict(image_np)
            predicted_class = np.argmax(prediction, axis=1) # Get the index of the highest probability class

            st.write(f"Predicted Class: {predicted_class[0]}")

            # Get the actual letter using the label map
            actual_label = label_map[predicted_class[0]]

            st.write(f"Predicted Letter: {actual_label}")
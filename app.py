import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Function to perform OCR on the uploaded image
def perform_ocr(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform OCR using pytesseract
    results = pytesseract.image_to_string(gray_image)
    return results

def main():
    st.title("OCR Master")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))

        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Perform OCR'):
            # Perform OCR
            ocr_results = perform_ocr(image.copy())

            # Display the extracted text
            st.subheader("Extracted Text:")
            st.write(ocr_results)

if __name__ == "__main__":
    main()

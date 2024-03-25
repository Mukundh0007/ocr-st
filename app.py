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
    results = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)
    return results

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    for i in range(len(results["text"])):
        # Extract bounding box coordinates
        x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]
        # Draw bounding box rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main():
    st.title("OCR Master")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))

        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Perform OCR'):
            # Perform OCR
            ocr_results = perform_ocr(image.copy())

            # Draw bounding boxes
            image_with_boxes = draw_boxes(image.copy(), ocr_results)

            # Display the image with bounding boxes
            st.image(image_with_boxes, caption='Image with Bounding Boxes', use_column_width=True)

            # Extract text from OCR results
            extracted_text = " ".join([text for text in ocr_results["text"] if text.strip()])

            # Display the extracted text
            st.subheader("Extracted Text:")
            st.write(extracted_text)

if __name__ == "__main__":
    main()

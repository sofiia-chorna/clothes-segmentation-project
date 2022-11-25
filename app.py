import json
import streamlit as st
from PIL import Image
import torch

from config import config
from image_processor import ImageRunner
import cv2
import numpy as np

# Retrieves list of available images given the current selections
@st.cache()
def load_image_from_storage(
        image_name: str
) -> str:
    image = Image.open(image_name)
    return image


# Retrieves list of available images given the current selections
@st.cache()
def load_list_of_images_available(
        all_image_files: dict,
        image_files_dtype: str,
) -> list:
    list_of_files = all_image_files.get(image_files_dtype)
    return list_of_files


# Retrieves JSON document outing the S3 file structure
@st.cache()
def load_s3_file_structure(path: str = 'all_image_files.json') -> dict:
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    all_image_files = load_s3_file_structure()
    n_classes = len(config['all_classes'])
    activation = 'sigmoid'
    model = torch.load(config['models'], map_location=torch.device('cpu'))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    st.header("Segment clothes in images")
    st.write("Choose any image and get corresponding clothes segmentation:")

    uploaded_file = st.file_uploader("Choose an image...")
    img = None
    if uploaded_file is not None:  # if user uploaded file
        st.image(uploaded_file, caption='Input Image', use_column_width=True)
        img = Image.open(uploaded_file)
    else:
        st.sidebar.header("Examples")
        available_images = load_list_of_images_available(all_image_files, 'samples')
        image_name = st.sidebar.selectbox("Image Name", available_images)
        path = './images/' + image_name
        img = load_image_from_storage(path)

        st.write("Here is the image you've selected:")
        resized_image = img.resize((362, 562))
        st.image(resized_image)

    m = np.array(img)
    m = cv2.resize(m, (362, 562), interpolation=cv2.INTER_AREA)

    with st.spinner('Processing the image...'):
        result_file = ImageRunner(model, m).process_image()

    st.image(result_file, caption='Clothes Segmentation', use_column_width=True)

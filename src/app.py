import json
import torch
import cv2
import streamlit as st
from PIL import Image
import numpy as np

from config import config

from image_processor import ImageRunner
from image_loader import ImageLoader

@st.cache(show_spinner=False)
def load_image_from_storage(
        image_name: str
) -> Image:
    """Retrieves list of available images given the current selections"""
    auth = json.load(open('auth.json'))
    loader = ImageLoader(auth['CLIENT_ID'], auth['CLIENT_SECRET'])
    return loader.get_image(image_name)


@st.cache()
def load_list_of_images_available(
        all_image_files: dict,
        image_files_dtype: str,
        clothes_type: str
) -> list:
    """Retrieves list of available images given the current selections"""
    types_dict = all_image_files.get(image_files_dtype)
    list_of_files = types_dict.get(clothes_type.lower())
    return list_of_files


@st.cache()
def load_s3_file_structure(path: str = 'all_image_files.json') -> dict:
    """Retrieves JSON document outing the S3 file structure"""
    with open(path, "r") as f:
        return json.load(f)


if __name__ == '__main__':
    all_image_files = load_s3_file_structure()
    n_classes = len(config['all_classes'])
    activation = 'sigmoid'
    model = torch.load(config['model'], map_location=torch.device('cpu'))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    types_of_clothes = sorted(list(all_image_files['test'].keys()))
    types_of_clothes = [clothes.title() for clothes in types_of_clothes]

    st.title('Welcome to Clothes Segmentation Project!')
    instructions = """
            Either upload your own image or select from
            the sidebar to get a preconfigured image.
            The image you select or upload will be fed
            through the Deep Neural Network in real-time
            and the output will be displayed to the screen.
            """
    st.write(instructions)

    uploaded_file = st.file_uploader("Choose an image...")
    img = None
    if uploaded_file is not None:  # if user uploaded file
        st.image(uploaded_file, caption='Input Image', use_column_width=True)
        img = Image.open(uploaded_file)
    else:
        st.sidebar.header("Examples")
        selected_types = st.sidebar.selectbox("Clothes Type", types_of_clothes)
        available_images = load_list_of_images_available(all_image_files, 'test', selected_types.upper())
        sidebar_images = ["Image {}".format(i) for i, _x in enumerate(available_images)]
        image_name = st.sidebar.selectbox("Image Name", sidebar_images)
        if image_name is None:
            image_name = available_images.upper()
        else:
            index = sidebar_images.index(image_name)
            image_name = available_images[index]

        with st.spinner('Loading the image...'):
            img = load_image_from_storage(image_name)

        st.title("Here is the image you've selected:")
        resized_image = img.resize((336, 562))
        st.image(resized_image, caption='Input Image')

    m = np.array(img)
    m = cv2.resize(m, (362, 562), interpolation=cv2.INTER_AREA)

    with st.spinner('Processing the image...'):
        result_file = ImageRunner(model, m).process_image()

    st.title(f"Here is the result of segmentation:")
    img = Image.open(result_file)
    resized_image = img.resize((336, 562))
    st.image(resized_image, caption='Clothes Segmentation')
    with open(result_file, "rb") as file:
        btn = st.download_button(
            label="Download result",
            data=file,
            file_name="segmentation_result.png",
            mime="image/png"
        )

import json
import streamlit as st
from segmentator import Segmentator
from PIL import Image

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

    st.header("Segment clothes in images")
    st.write("Choose any image and get corresponding clothes segmentation:")

    uploaded_file = st.file_uploader("Choose an image...")

    segmentator = None
    if uploaded_file is not None:  # if user uploaded file
        st.image(uploaded_file, caption='Input Image', use_column_width=True)

        segmentator = Segmentator(uploaded_file)

    else:
        st.sidebar.header("Examples")
        available_images = load_list_of_images_available(all_image_files, 'samples')
        image_name = st.sidebar.selectbox("Image Name", available_images)
        path = './images/' + image_name
        img = load_image_from_storage(path)

        st.write("Here is the image you've selected:")
        resized_image = img.resize((362, 562))
        st.image(resized_image)

        segmentator = Segmentator(path)

    result_file = segmentator.get_dress()
    st.image(result_file, caption='Clothes Segmentation', use_column_width=True)

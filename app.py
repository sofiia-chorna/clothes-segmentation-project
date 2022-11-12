import streamlit as st
from segmentator import Segmentator

if __name__ == '__main__':
    st.header("Segment clothes in images")
    st.write("Choose any image and get corresponding clothes segmentation:")

    uploaded_file = st.file_uploader("Choose an image...")

    if uploaded_file is not None:  # if user uploaded file
        st.image(uploaded_file, caption='Input Image', use_column_width=True)

        segmentator = Segmentator(uploaded_file)
        result_file = segmentator.get_dress()

        st.image(result_file, caption='Clothes Segmentation', use_column_width=True)

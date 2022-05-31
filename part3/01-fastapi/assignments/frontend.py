import io
import os
from pathlib import Path
import numpy as np
import requests
from PIL import Image

import streamlit as st

# SETTING PAGE CONFIG TO WIDE MODE
# ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

st.set_page_config(layout="wide")

root_password = 'password'


def main():
    st.title("Text2Image Model")
    uploaded_text = st.text_input("Fill in the blank!")

    if uploaded_text:
        # st.write("Translating...")

        st.write("Generating...")
        
        image_array = requests.post("http://101.101.208.118:30002/order", text=uploaded_text)
        image = Image.open(Image.fromarray((image_array*255).astype(np.uint8)))
        
        st.image(image, caption='Generated Image')
    else:
        st.write(f'Fill in the blank FIRST!')
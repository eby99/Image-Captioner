import os
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import gdown
from model import Net
from utils import ConfigL, ConfigS, download_weights




# File download from Google Drive
file_id = '1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF'
destination = 'weights/small/model.pt'
url = f'https://drive.google.com/uc?id={file_id}'

# Ensure the directory exists
os.makedirs('weights/small', exist_ok=True)

# Check if the file already exists
if not os.path.exists(destination):
    st.write("Downloading model weights...")
    gdown.download(url, destination, quiet=False)
else:
    st.write("Model weights already downloaded.")

# Load the model after downloading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(destination, map_location=device)
# Your model loading code continues...
# Streamlit application
st.title("Image Caption Generator")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Model configuration
    model_size = st.selectbox("Select Model Size", options=["S", "L"])
    # temperature = st.slider("Select Temperature", min_value=0.0, max_value=2.0, value=1.0)

    if st.button("Generate Caption"):
        config = ConfigL() if model_size.upper() == "L" else ConfigS()

        # Set seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True

        is_cuda = torch.cuda.is_available()
        device = "cuda" if is_cuda else "cpu"

        ckp_path = os.path.join(config.weights_dir, "model.pt")

        if not os.path.isfile(ckp_path):
            download_weights(ckp_path, model_size)

        model = Net(
            clip_model=config.clip_model,
            text_model=config.text_model,
            ep_len=config.ep_len,
            num_layers=config.num_layers,
            n_heads=config.n_heads,
            forward_expansion=config.forward_expansion,
            dropout=config.dropout,
            max_len=config.max_len,
            device=device,
        )

        checkpoint = torch.load(ckp_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        with torch.no_grad():
            caption, _ = model(img)

        st.write(f'Generated Caption: "{caption}"')

        # Save the image with the caption
        plt.imshow(img)
        plt.title(caption)
        plt.axis("off")

        img_save_path = f'{os.path.splitext(uploaded_file.name)[0]}-captioned.jpg'
        plt.savefig(img_save_path, bbox_inches="tight")
        plt.clf()
        plt.close()

        # Display the saved image
        st.image(img_save_path, caption='Captioned Image.', use_column_width=True)
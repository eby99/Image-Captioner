import os
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import gdown

from model import Net
from utils import ConfigL, ConfigS

# Google Drive model weights dictionary
MODEL_WEIGHTS = {
    "L": "1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG",
    "S": "1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF",
}

def download_weights(checkpoint_fpath, model_size="L"):
    """ Downloads weights from Google Drive. """
    download_id = MODEL_WEIGHTS[model_size.strip().upper()]
    gdown.download(
        f"https://drive.google.com/uc?id={download_id}", checkpoint_fpath, quiet=False
    )

# Streamlit application
st.title("Image Caption Generator")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Attempt to open the image
        img = Image.open(uploaded_file)
        img = img.convert("RGB")  # Convert to RGB to ensure compatibility
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        # Model configuration
        model_size = st.selectbox("Select Model Size", options=["SMALL", "LARGE"])

        if st.button("Generate Caption"):
            try:
                config = ConfigL() if model_size.upper() == "LARGE" else ConfigS()
            except Exception as e:
                st.error(f"Error initializing configuration: {e}")
                st.stop()  # Stop further execution if configuration fails

            # Ensure weights directory exists
            weights_dir = config.weights_dir
            if not os.path.exists(weights_dir):
                os.makedirs(weights_dir)

            # Set seed
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            torch.backends.cudnn.deterministic = True

            is_cuda = torch.cuda.is_available()
            device = "cuda" if is_cuda else "cpu"

            ckp_path = os.path.join(weights_dir, "model.pt")

            # Download weights at runtime if not present
            if not os.path.isfile(ckp_path):
                st.write("Downloading model weights...")
                download_weights(ckp_path, model_size)
                st.success("Model weights downloaded successfully!")

            # Load the model
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

            try:
                checkpoint = torch.load(ckp_path, map_location=device)
                model.load_state_dict(checkpoint)
                model.eval()

                with st.spinner('Generating caption...'):
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

            except Exception as e:
                st.error(f"Error loading model: {e}")

    except Exception as e:
        st.error(f"Error loading image: {e}")

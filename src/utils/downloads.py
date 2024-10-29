import os
import gdown

MODEL_WEIGHTS = {
    "L": "1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG",
    "S": "1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF",
}

def download_weights(checkpoint_fpath, model_size="L"):
    """
    Downloads weights from Google Drive.
    """
    if os.path.exists(checkpoint_fpath):
        print(f"File {checkpoint_fpath} already exists. Skipping download.")
        return

    download_id = MODEL_WEIGHTS[model_size.strip().upper()]
    gdown.download(
        f"https://drive.google.com/uc?id={download_id}", checkpoint_fpath, quiet=False
    )

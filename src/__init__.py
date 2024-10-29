"""
Utility functions and configurations for the project.
"""

import gdown

MODEL_WEIGHTS = {
    "L": "1Gh32arzhW06C1ZJyzcJSSfdJDi3RgWoG",
    "S": "1pSQruQyg8KJq6VmzhMLFbT_VaHJMdlWF",
}

def download_weights(checkpoint_fpath, model_size="L"):
    """
    Downloads weights from Google Drive.
    """

    download_id = MODEL_WEIGHTS[model_size.strip().upper()]

    gdown.download(
        f"https://drive.google.com/uc?id={download_id}", checkpoint_fpath, quiet=False
    )

class ConfigS:
    def __init__(self):
        self.seed = 42
        self.weights_dir = "weights/small"  # Ensure this path exists or will be created
        self.clip_model = "clip_model_name"  # Add actual model names
        self.text_model = "text_model_name"
        self.ep_len = 20
        self.num_layers = 6
        self.n_heads = 8
        self.forward_expansion = 4
        self.dropout = 0.1
        self.max_len = 75

class ConfigL:
    def __init__(self):
        self.seed = 42
        self.weights_dir = "weights/large"  # Ensure this path exists or will be created
        self.clip_model = "clip_model_large_name"  # Add actual model names
        self.text_model = "text_model_large_name"
        self.ep_len = 20
        self.num_layers = 12
        self.n_heads = 12
        self.forward_expansion = 4
        self.dropout = 0.1
        self.max_len = 75

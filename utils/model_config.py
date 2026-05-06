import os
import urllib.request

# 1. Model configurations and URLs
MODEL_CONFIGS = {
    "RDN_best": {
        "url": "https://github.com/GiulioFede/QuADA-GS/releases/download/model_weights/RDN_based_best.ckpt",
        "filename": "RDN_based_best.ckpt",
        "architecture": "QuADA_GS_v2" # Uses RDN
    },
    "RDN_classic": {
        "url": "https://LINK_PLACEHOLDER/RDN_based_classic.ckpt", # TODO: Insert actual link
        "filename": "RDN_based_classic.ckpt",
        "architecture": "QuADA_GS_v2" # Uses RDN
    },
    "EDSR_best": {
        "url": "https://LINK_PLACEHOLDER/EDSR_based_best.ckpt", # TODO: Insert actual link
        "filename": "EDSR_based_best.ckpt",
        "architecture": "QuADA_GS_v3" # Uses EDSR
    }
}

# 2. Auto-download function
def get_and_download_model(model_choice):
    config = MODEL_CONFIGS[model_choice]
    weight_dir = "weights"
    weight_path = os.path.join(weight_dir, config["filename"])

    if not os.path.exists(weight_path):
        print(f"[*] Weights for '{model_choice}' not found locally.")
        print(f"[*] Starting download from GitHub (this may take a few minutes)...")
        os.makedirs(weight_dir, exist_ok=True)
        try:
            urllib.request.urlretrieve(config["url"], weight_path)
            print("[*] Download completed successfully!")
        except Exception as e:
            print(f"[!] Error during download: {e}")
            print(f"[!] Please download the file manually from {config['url']}")
            print(f"[!] and save it to: {weight_path}")
            exit(1)
            
    return weight_path, config["architecture"]
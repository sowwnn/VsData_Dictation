import os
import requests
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def download_file(url, destination):
    """
    Downloads a file from a direct URL to a specified destination, with a progress bar
    and content type validation.
    """
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            
            # --- Content-Type Check ---
            content_type = r.headers.get('content-type')
            if 'text/html' in content_type:
                logging.error("Download failed: The server returned an HTML page instead of a file.")
                logging.error("This can happen if the link is expired or requires a login.")
                logging.error("Please find an updated, direct download link for the model.")
                return False

            total_size = int(r.headers.get('content-length', 0))
            
            with open(destination, "wb") as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=os.path.basename(destination)) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred during download: {e}")
        return False


def download_medsam_model(destination_dir="server/model"):
    """
    Downloads the MedSAM model checkpoint from a reliable, direct source.
    """
    # URL from the official MedSAM repository, less likely to fail than Google Drive.
    model_url = "https://www.dropbox.com/scl/fi/bd7243j9txl7bb502kea9/medsam_vit_b.pth?rlkey=my2w5045s5vtrqsonde52n56f&dl=1"
    
    # Ensure the destination directory exists
    expanded_dir = os.path.expanduser(destination_dir)
    os.makedirs(expanded_dir, exist_ok=True)
    
    model_path = os.path.join(expanded_dir, "medsam_vit_b.pth")
    
    if os.path.exists(model_path):
        # You could add a file size check here for more robustness
        logging.info(f"MedSAM model already exists at {model_path}")
        return

    logging.info(f"Downloading MedSAM model from new source to {model_path}...")
    
    if download_file(model_url, model_path):
        logging.info("MedSAM model downloaded successfully.")
    else:
        logging.error("Failed to download MedSAM model.")
        logging.error("Please check your internet connection or the model URL and try again.")
        if os.path.exists(model_path): # Clean up partial/failed download
            os.remove(model_path)


if __name__ == "__main__":
    download_medsam_model()

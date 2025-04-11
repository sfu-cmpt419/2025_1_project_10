import gdown
import zipfile
import os

def download_and_extract():
    # Google Drive file ID
    file_id = "18inDeehM-gXOWzvjDgDKs47XjHYSsxSq"
    # Construct direct download link
    url = f"https://drive.google.com/uc?id={file_id}"

    # Where to save the downloaded zip
    output_zip = "data.zip"
    data_folder = "data"

    # Step 1: Download
    if not os.path.exists(output_zip):
        print(f"Downloading dataset from Google Drive...")
        gdown.download(url, output=output_zip, quiet=False)
        print("Download completed!")
    else:
        print(f"{output_zip} already exists. Skipping download.")

    # Step 2: Extract
    if not os.path.exists(data_folder):
        print(f"Extracting {output_zip}...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(data_folder)
        print("Extraction completed!")
    else:
        print(f"{data_folder}/ already exists. Skipping extraction.")

if __name__ == "__main__":
    download_and_extract()

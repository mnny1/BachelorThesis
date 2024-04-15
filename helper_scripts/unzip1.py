import os
from zipfile import ZipFile

def unzip_all_files(root_folder):
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                with ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(zip_path)

if __name__ == "__main__":
    # Change this to your root folder path
    root_folder = "data/train_val_v1_coco_format/images/train"
    unzip_all_files(root_folder)

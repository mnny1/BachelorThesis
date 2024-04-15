import os, zipfile

"""
script for unzipping trays

"""



root_dir = "data/train_val_cam/train_set/jpegs"

for root, dirs, files in os.walk(root_dir):
    for file in files:

        # Check if the file is a ZIP archive
        if file.endswith(".zip"):
            zip_file_path = os.path.join(root, file)
            print(zip_file_path)
            
            # Create a directory to extract the contents of the ZIP file
            # extraction_dir = os.path.join(root, os.path.splitext(file)[0])
            # os.makedirs(extraction_dir, exist_ok=True)

            # Extract the ZIP file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(root)

                # remove zip file
                os.remove(zip_file_path)

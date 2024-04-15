import os

"""
script for deleting undesired images within the train set, like img with unidentified Weed species" 
"""


root_folder = "data/train_val_cam/train_set/jpegs"

with open("data/train_val_cam/undesired_entries_filename.txt", "r") as f:
    undesired_images = [name.strip() + ".jpeg" for name in f.read().split()]
    
for filename in undesired_images:
    file_path = os.path.join(root_folder, filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"File not found: {file_path}")
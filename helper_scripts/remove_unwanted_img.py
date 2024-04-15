import os

def delete_images_from_list(file_path, image_folder):

  with open(file_path, 'r') as f:
    # Read the entire line
    filepaths = f.readline().strip().split(" ")

  for file in filepaths:
    image_name = f"{file.split('/')[2]}.jpeg"  # Assuming ".jpeg" extension
    image_path = os.path.join(image_folder, image_name)

    # Check if file exists before deleting
    if os.path.exists(image_path):
      os.remove(image_path)
      print(f"Deleted: {image_name}")
    else:
      print(f"File not found: {image_path}")
# Replace these with your actual file paths
text_file_path = "data/train_val_cam/undesired_entries_filename.txt"
image_folder_path = "data/train_val_v1_coco_format/images/train"

delete_images_from_list(text_file_path, image_folder_path)

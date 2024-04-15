import os
import shutil

def move_images_to_root(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(root_folder, file)
                shutil.move(src_path, dest_path)

if __name__ == "__main__":
    root_folder = "data/real_data_v1_coco_format/images/fodder"  # Change this to your actual root folder path
    move_images_to_root(root_folder)
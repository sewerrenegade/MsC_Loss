import os
import shutil

def move_small_folders(src_folder, dest_folder, min_images=10):
    """
    Iterates through the subfolders of 'src_folder' and moves those containing fewer than
    'min_images' images to 'dest_folder'.
    
    Args:
        src_folder (str): Path to the source folder containing subfolders.
        dest_folder (str): Path to the destination folder where small folders will be moved.
        min_images (int): Minimum number of images required in a folder to retain it in the source.
    """
    # Ensure destination folder exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate through subdirectories (folders) in the source folder
    for subfolder in os.listdir(src_folder):
        subfolder_path = os.path.join(src_folder, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            image_count = 0
            # Iterate through the files in the subfolder and count images (e.g., jpg, png)
            for file in os.listdir(subfolder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):  # Customize extensions if needed
                    image_count += 1

            # If the number of images is smaller than min_images, move the folder to dest_folder
            if image_count < min_images:
                dest_subfolder_path = os.path.join(dest_folder, subfolder)
                print(f"Moving folder '{subfolder}' with {image_count} images to {dest_folder}")
                shutil.move(subfolder_path, dest_subfolder_path)

# Example usage
src_folder = "/lustre/groups/shared/users/milad.bassil/datasets/imagenet_a/data_imagenet_a"  # Replace with your source folder path
dest_folder = "/lustre/groups/shared/users/milad.bassil/datasets/imagenet_a/data_imagenet_a_small"  # Replace with your destination folder path
min_images = 35  # Minimum number of images in a folder
move_small_folders(src_folder, dest_folder, min_images)

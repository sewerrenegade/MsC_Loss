
import pandas as pd
import os
mapping_file = "data_imagenet_a/class_name_mapping.csv"
data_dir = "data_imagenet_a"
def _remap_folders():
    """Rename subfolders based on class_name_mapping.csv."""
    mapping = pd.read_csv(mapping_file).set_index("ID")["Name"].to_dict()
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder in mapping:
            new_name = mapping[subfolder]
            new_path = os.path.join(data_dir, new_name)
            if not os.path.exists(new_path):
                os.rename(subfolder_path, new_path)
                print(f"remaoming {subfolder_path} to {new_name}")
            
            
            
if __name__ == "__main__":
    _remap_folders()
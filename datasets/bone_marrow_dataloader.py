import os
import numpy as np
import pandas as pd
import torch
SEED = 42
torch.manual_seed(SEED)
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from pytorch_lightning import LightningDataModule
from PIL import Image
from sklearn.preprocessing import LabelEncoder

SEED = 42
torch.manual_seed(SEED)

AUTO_ENCODER_PROPORTION_OF_DATA = 0.5
CLASSIFIER_PROPORTION_OF_DATA = 1 - AUTO_ENCODER_PROPORTION_OF_DATA
AUTO_ENCODER_SPLIT = (0.6, 0.2, 0.2)
CLASSIFIER_SPLIT = (0.05, 0.05, 0.9)
DATASET_TARGETS = ['ae', 'classifier']

class BoneMarrowDataset(Dataset):
    def __init__(self, csv_file="bone_marrow/bm_test.csv", transform=None,min_class_count = 10):
        self.data = pd.read_csv(csv_file, header=None, names=["image_path", "label"])

        # Filter classes with at least 'n' instances
        self.min_class_count = min_class_count
        self.data = self.data.groupby("label").filter(lambda x: len(x) >= self.min_class_count)

        # Reset index if needed
        self.data.reset_index(drop=True, inplace=True)
                # Encode labels as integers
        self.label_encoder = LabelEncoder()
        self.data["target"] = self.label_encoder.fit_transform(self.data["label"])
        self.num_classes = len(self.label_encoder.classes_)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, target = self.data.iloc[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, target

class BoneMarrowImageDataModule(LightningDataModule):
    def __init__(self, data_dir="bone_marrow/bm_test.csv", dataset_target="classifier", batch_size=32, num_workers=4, stratify=False):
        super().__init__()
        assert dataset_target in DATASET_TARGETS
        self.dataset_target = dataset_target
        self.csv_file = data_dir
        self.ae_proportion = AUTO_ENCODER_PROPORTION_OF_DATA
        self.ae_splits = AUTO_ENCODER_SPLIT
        self.classifier_splits = CLASSIFIER_SPLIT
        self.classifier_proportion = CLASSIFIER_PROPORTION_OF_DATA
        self.stratify = stratify
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.Resize((144, 144)),
            transforms.ToTensor()
        ])
        self.full_dataset  = BoneMarrowDataset(self.csv_file, transform=self.transform,min_class_count=10)
        self.nb_classes = self.full_dataset.num_classes
        self.dataset_type = "single_cell"
        
    def setup(self, stage=None):
        if stage == "fit":
            if self.stratify:
                print("Running startified setup of dataset")
                self.stratified_setup(stage)
            else:
                print("Running UNstartified setup of dataset")
                self.unstratisfied_setup(stage)
        elif stage == "test":
            print("Test dataloader already setup during training setup")              
    def unstratisfied_setup(self, stage=None):
        """
        Prepares the dataset splits for training, validation, and testing.
        """
        # Load the full dataset
        full_dataset = self.full_dataset
        total_samples = len(full_dataset)
        ae_data_size = int(self.ae_proportion * total_samples)
        classifier_data_size = int(self.classifier_proportion * total_samples)
        trash_size = total_samples - ae_data_size - classifier_data_size
        ae_dataset, classifier_dataset, _ = random_split(full_dataset, [ae_data_size, classifier_data_size, trash_size])

        if self.dataset_target == "ae":
            split = self.ae_splits
            dataset = ae_dataset
        elif self.dataset_target == "classifier":
            split = self.classifier_splits
            dataset = classifier_dataset
            
        train_size = int(len(dataset) * split[0])
        val_size = int(len(dataset) * split[1])
        test_size = len(dataset) - train_size - val_size
        self.train_set, self.val_set, self.test_set = random_split(
            dataset, [train_size, val_size, test_size]
        )    
    def stratified_setup(self,stage):
        from collections import Counter
        full_dataset = self.full_dataset
        total_samples = len(full_dataset)
        ae_data_size = int(self.ae_proportion * total_samples)
        classifier_data_size = int(self.classifier_proportion * total_samples)
        trash_size = total_samples - ae_data_size - classifier_data_size
        targets = full_dataset.data["target"].values
        class_counts = full_dataset.data["label"].value_counts()
        print(f"class counts: {class_counts}")
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=classifier_data_size, random_state=SEED)
        ae_idx, classifier_idx = next(sss.split(np.zeros(total_samples), targets))

        # Convert indices to datasets
        ae_dataset = Subset(full_dataset, ae_idx)
        classifier_dataset = Subset(full_dataset, classifier_idx)

        # Choose dataset based on target
        dataset = ae_dataset if self.dataset_target == "ae" else classifier_dataset
        dataset_indices = ae_idx if self.dataset_target == "ae" else classifier_idx  # Use correct indices
        dataset_targets = targets[dataset_indices]  # Extract targets for this dataset

        # Get train/val/test split sizes
        split = self.ae_splits if self.dataset_target == "ae" else self.classifier_splits
        train_size = int(len(dataset) * split[0])
        val_size = int(len(dataset) * split[1])
        test_size = len(dataset) - train_size - val_size

        # Stratified train/val/test split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=SEED)
        train_idx, remaining_idx = next(sss.split(np.zeros(len(dataset)), dataset_targets))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        val_idx, test_idx = next(sss.split(np.zeros(len(remaining_idx)), dataset_targets[remaining_idx]))

        # Convert to Subsets
        self.train_set = Subset(dataset, train_idx)
        self.val_set = Subset(dataset, [remaining_idx[i] for i in val_idx])
        self.test_set = Subset(dataset, [remaining_idx[i] for i in test_idx])
        def print_split_stats(name, subset):
            subset_indices = subset.indices  # Extract subset indices
            subset_labels = targets[dataset_indices][subset_indices]  # Get corresponding targets
            label_counts = dict(Counter(subset_labels))
            sorted_dict = {k: label_counts[k] for k in sorted(label_counts.keys())}
            print(f"{name} - Total: {len(subset)} | Class Distribution: {sorted_dict}")
        print(f"Total size of Acevedo {total_samples}")
        print_split_stats("Train", self.train_set)
        print_split_stats("Validation", self.val_set)
        print_split_stats("Test", self.test_set)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=(self.device == "cuda"))

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=(self.device == "cuda"))

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=(self.device == "cuda"))

if __name__ == "__main__":
    datamodule = BoneMarrowImageDataModule(dataset_target="classifier", batch_size=64, num_workers=2,stratify=True)
    datamodule.setup(stage="fit")

    # Example usage
    train_loader = datamodule.train_dataloader()
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}, Batch labels: {labels}")
       
       
       
       
       
        

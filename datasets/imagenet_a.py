import os
from torchvision import datasets, transforms
import torch
SEED = 42
torch.manual_seed(SEED)
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pytorch_lightning import LightningDataModule
from PIL import Image

AUTO_ENCODER_PROPORTION_OF_DATA = 0.85
CLASSIFIER_PROPORTION_OF_DATA = 1 - AUTO_ENCODER_PROPORTION_OF_DATA
AUTO_ENCODER_SPLIT = (0.6, 0.2, 0.2)
CLASSIFIER_SPLIT = (0.05, 0.05, 0.9)
DATASET_TARGETS = ['ae','classifier']
class ImageNetADataModule(LightningDataModule):
    def __init__(self, dataset_target = "classifier", data_dir="data_imagenet_a", batch_size=32, num_workers=4,stratify = False):
        """
        Args:
            data_dir (str): Path to the main data folder.
            part_a_proportion (float): Proportion of data assigned to Part A.
            part_a_splits (tuple): Proportions of training, validation, and testing for Part A.
            part_b_splits (tuple): Proportions of training, validation, and testing for Part B.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
        """
        super().__init__()
        assert dataset_target in DATASET_TARGETS
        self.dataset_target = dataset_target
        self.data_dir = data_dir
        self.ae_proportion = AUTO_ENCODER_PROPORTION_OF_DATA
        self.ae_splits = AUTO_ENCODER_SPLIT
        self.classifier_splits = CLASSIFIER_SPLIT
        self.classifier_proportion = CLASSIFIER_PROPORTION_OF_DATA
        self.startify = stratify
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((144, 144)),
            transforms.ToTensor()
            
        ])
        self.full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        self.filter_classes_by_instance_count(10)
        self.nb_classes = len(self.full_dataset.class_to_idx)
        self.dataset_type = "image_net"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Transforms to apply to the images
        

        self.part_a_data = None
        self.part_b_data = None
        
    def filter_classes_by_instance_count(self, min_instances):
        # Count the number of instances per class
        dataset = self.full_dataset
        class_counts = {class_idx: 0 for class_idx in dataset.class_to_idx.values()}
        
        for _, label in dataset.samples:
            class_counts[label] += 1

        # Keep only classes that have at least `min_instances` samples
        valid_classes = {class_idx for class_idx, count in class_counts.items() if count >= min_instances}
        
        # Filter dataset samples
        filtered_samples = [(path, label) for path, label in dataset.samples if label in valid_classes]
        
        # Create new ImageFolder dataset
        dataset.samples = filtered_samples
        dataset.targets = [label for _, label in filtered_samples]
        
        self.full_dataset = dataset
    def setup(self, stage = None):
        if stage == "fit":
            if self.startify:
                print("Running startified setup of dataset")
                self.startified_setup(stage)
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
        assert (self.classifier_proportion-self.ae_proportion) <= 1 #make sure no data leak occured
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
        
    def startified_setup(self, stage=None):
        """
        Prepares the dataset splits for training, validation, and testing in a deterministic stratified way.
        """
        from collections import Counter
        RANDOM_SEED = 42  # Set a fixed seed for deterministic splits

        # Load the full dataset
        assert (self.classifier_proportion + self.ae_proportion) <= 1, "Data leakage detected!"

        full_dataset = self.full_dataset 
        total_samples = len(full_dataset)
        targets = np.array(full_dataset.targets)  # Extract class labels

        ae_data_size = int(self.ae_proportion * total_samples)
        classifier_data_size = int(self.classifier_proportion * total_samples)

        # Stratify into AE and remaining data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=classifier_data_size, random_state=RANDOM_SEED)
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
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=RANDOM_SEED)
        train_idx, remaining_idx = next(sss.split(np.zeros(len(dataset)), dataset_targets))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)
        val_idx, test_idx = next(sss.split(np.zeros(len(remaining_idx)), dataset_targets[remaining_idx]))

        # Convert to Subsets
        self.train_set = Subset(dataset, train_idx)
        self.val_set = Subset(dataset, [remaining_idx[i] for i in val_idx])
        self.test_set = Subset(dataset, [remaining_idx[i] for i in test_idx])

        # Helper function to print class distributions
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

    def get_pic_path(self,dataset,idx):
        while hasattr(dataset,"dataset"):
            idx = dataset.indices[idx]
            dataset = dataset.dataset
        return dataset.imgs[idx]
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,persistent_workers=True,
                          num_workers=self.num_workers, shuffle=True, pin_memory=(self.device == "cuda"),
                          **({"pin_memory_device": "cuda"} if self.device == "cuda" else {}))

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,persistent_workers=True,
                          num_workers=self.num_workers, shuffle=False, pin_memory=(self.device == "cuda"),
                          **({"pin_memory_device": "cuda"} if self.device == "cuda" else {}))

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, pin_memory=(self.device == "cuda"),
                          **({"pin_memory_device": "cuda"} if self.device == "cuda" else {}))


if __name__ == "__main__":
    datamodule = DualAcevedoImageDataModule(dataset_target="classifier", batch_size=64, num_workers=2)
    datamodule.setup()

    # Example usage
    train_loader = datamodule.train_dataloader()
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}, Batch labels: {labels}")
       
       
       
       
       
        

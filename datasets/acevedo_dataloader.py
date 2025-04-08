import os
from typing import Counter
from torchvision import datasets, transforms
import torch

SEED = 42
torch.manual_seed(SEED)
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from pytorch_lightning import LightningDataModule
from PIL import Image

AUTO_ENCODER_PROPORTION_OF_DATA = 0.85
CLASSIFIER_PROPORTION_OF_DATA = 1 - AUTO_ENCODER_PROPORTION_OF_DATA
AUTO_ENCODER_SPLIT = (0.6, 0.2, 0.2)
CLASSIFIER_SPLIT = (0.05, 0.05, 0.9)
DATASET_TARGETS = ['ae','classifier']
class DualAcevedoImageDataModule(LightningDataModule):
    def __init__(self, dataset_target = "classifier", data_dir=os.path.join("Acevedo", "processed_images_144"), base_data_dir = "",batch_size=32, num_workers=4,stratify = False,k_fold = -1,fold_nb = 0):
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
        self.final_data_dir = data_dir
        self.base_dir = base_data_dir
        self.data_dir = os.path.join(self.base_dir,self.final_data_dir)
        self.ae_proportion = AUTO_ENCODER_PROPORTION_OF_DATA
        self.ae_splits = AUTO_ENCODER_SPLIT
        self.classifier_splits = CLASSIFIER_SPLIT
        self.classifier_proportion = CLASSIFIER_PROPORTION_OF_DATA
        self.startify = stratify
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nb_classes = 8
        self.dataset_type = "single_cell"
        self.full_dataset = None
        self.k_fold = k_fold
        self.fold_nb = fold_nb
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Transforms to apply to the images
        self.transform = transforms.Compose([
            transforms.Resize((144, 144)),
            transforms.ToTensor()
            
        ])
        
        self.part_a_data = None
        self.part_b_data = None
        self.full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        self.class_weighting = DualAcevedoImageDataModule.calculate_class_weights(self.full_dataset)
    
    @staticmethod
    def calculate_class_weights(dataset):
        # Get the class labels (targets) from the dataset
        targets = dataset.targets  # ImageFolder stores the labels in `targets`
        
        # Count the frequency of each class
        label_counts = dict(Counter(targets))
        
        # Calculate class weights (inverse of the class frequencies)
        total_samples = len(targets)
        class_weights = {k: total_samples / count for k, count in label_counts.items()}
        
        # Convert the class weights to a tensor
        weights_tensor = torch.tensor([class_weights[label] for label in sorted(label_counts.keys())], dtype=torch.float)
    
        return weights_tensor
        
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
        full_dataset,total_samples,ae_data_size,classifier_data_size = self.full_dataset ,len(self.full_dataset),int(self.ae_proportion * len(self.full_dataset)),int(self.classifier_proportion * len(self.full_dataset)) 
        trash_size = total_samples - ae_data_size - classifier_data_size
        ae_dataset, classifier_dataset, _ = random_split(full_dataset, [ae_data_size, classifier_data_size, trash_size],generator=torch.Generator().manual_seed(SEED))

        split,dataset = (self.ae_splits,ae_dataset) if self.dataset_target == "ae" else (self.classifier_splits,classifier_dataset)
            
        train_size,val_size = int(len(dataset) * split[0]),int(len(dataset) * split[1])
        test_size = len(dataset) - train_size - val_size
        
        if self.k_fold < 2:
            self.train_set, self.val_set, self.test_set = random_split(
                dataset, [train_size, val_size, test_size],generator=torch.Generator().manual_seed(SEED)
            )
        else:
            train_val , test = random_split(
                dataset, [train_size + val_size, test_size],generator=torch.Generator().manual_seed(SEED)
            )
            train_val_indices = list(range(len(train_val)))
        
            # Normal KFold without stratification
            kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=SEED)

            # Generate all splits and return the requested fold
            for i, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
                if i == self.fold_nb:
                    train_subset,val_subset = Subset(train_val, train_idx),Subset(train_val, val_idx)
                    self.train_set, self.val_set, self.test_set = train_subset, val_subset, test
                    return   


    def startified_setup(self, stage=None):
        """
        Prepares the dataset splits for training, validation, and testing in a deterministic stratified way.
        """
        # Load the full dataset
        assert (self.classifier_proportion + self.ae_proportion) <= 1, "Data leakage detected!"

        full_dataset, total_samples,targets,classifier_data_size = self.full_dataset,len(self.full_dataset),np.array(self.full_dataset.targets),int(self.classifier_proportion * len(self.full_dataset))

        # split data between classifier and ae experiments
        sss = StratifiedShuffleSplit(n_splits=1, test_size=classifier_data_size, random_state=SEED)
        ae_idx, classifier_idx = next(sss.split(np.zeros(total_samples), targets))

        # Convert indices to datasets
        ae_dataset, classifier_dataset = Subset(full_dataset, ae_idx), Subset(full_dataset, classifier_idx)

        # Choose dataset based on target
        dataset,dataset_indices,split = (ae_dataset,ae_idx,self.ae_splits) if self.dataset_target == "ae" else (classifier_dataset,classifier_idx,self.classifier_splits)
        dataset_targets = targets[dataset_indices]  # Extract targets for this dataset

        train_size,val_size = int(len(dataset) * split[0]),int(len(dataset) * split[1])
        test_size = len(dataset) - train_size - val_size

        # Stratified train/val/test split
        sss = StratifiedShuffleSplit(n_splits=1, test_size= test_size, random_state=SEED)
        train_val_idx, test_idx = next(sss.split(np.zeros(len(dataset)), dataset_targets))
        self.test_set = Subset(dataset, test_idx)
        
        if self.k_fold < 2:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
            train_idx, val_idx = next(sss.split(np.zeros(len(train_val_idx)), dataset_targets[train_val_idx]))        
            self.train_set = Subset(dataset, [train_val_idx[i] for i in train_idx])
            self.val_set = Subset(dataset, [train_val_idx[i] for i in val_idx])
        else:
            skf = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=SEED)
    
            # Iterate through folds and select the desired split
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_val_idx)), dataset_targets[train_val_idx])):
                if fold_idx == self.fold_nb:  # Select the specific fold
                    self.train_set = Subset(dataset, [train_val_idx[i] for i in train_idx])
                    self.val_set = Subset(dataset, [train_val_idx[i] for i in val_idx])
                    break  # Stop once we find the correct split
        # Helper function to print class distributions
        
        print(f"Total size of ImagenetA {total_samples}")
        DualAcevedoImageDataModule.print_split_stats("Train", self.train_set,targets,dataset_indices)
        DualAcevedoImageDataModule.print_split_stats("Validation", self.val_set,targets,dataset_indices)
        DualAcevedoImageDataModule.print_split_stats("Test", self.test_set,targets,dataset_indices)
    @staticmethod
    def calculate_train_class_weighting(subset,targets,dataset_indices):
        from collections import Counter
        subset_indices = subset.indices  # Extract subset indices
        subset_labels = targets[dataset_indices][subset_indices]  # Get corresponding targets
        label_counts = dict(Counter(subset_labels))
        total_samples = len(subset_labels)
        class_weights = {k: total_samples / count for k, count in label_counts.items()}
        return class_weights
        
    @staticmethod
    def print_split_stats(name, subset,targets,dataset_indices):
        from collections import Counter
        subset_indices = subset.indices  # Extract subset indices
        subset_labels = targets[dataset_indices][subset_indices]  # Get corresponding targets
        label_counts = dict(Counter(subset_labels))
        sorted_dict = {k: label_counts[k] for k in sorted(label_counts.keys())}
        print(f"{name} - Total: {len(subset)} | Class Distribution: {sorted_dict}")
        

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
       
       
       
       
       
        

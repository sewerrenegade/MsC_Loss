import os
from torchvision import datasets, transforms
import torch
SEED = 42
torch.manual_seed(SEED)
from torch.utils.data import DataLoader, random_split, Subset
from pytorch_lightning import LightningDataModule
from PIL import Image

AUTO_ENCODER_PROPORTION_OF_DATA = 0.7
CLASSIFIER_PROPORTION_OF_DATA = 1 - AUTO_ENCODER_PROPORTION_OF_DATA
AUTO_ENCODER_SPLIT = (0.7, 0.2, 0.1)
CLASSIFIER_SPLIT = (0.7, 0.2, 0.1)
DATASET_TARGETS = ['ae','classifier']
class DualAcevedoImageDataModule(LightningDataModule):
    def __init__(self, dataset_target = "ae", data_dir=os.path.join("Acevedo", "processed_images_144"), batch_size=32, num_workers=4):
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Transforms to apply to the images
        self.transform = transforms.Compose([
            transforms.Resize((144, 144)),
            transforms.ToTensor()
            
        ])

        self.part_a_data = None
        self.part_b_data = None

    def setup(self, stage=None):
        """
        Prepares the dataset splits for training, validation, and testing.
        """
        # Load the full dataset
        assert (self.classifier_proportion-self.ae_proportion) <= 1 #make sure no data leak occured
        full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
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
        x = self.train_set[0]
        y = dataset[self.train_set.indices[0]]
        z = full_dataset[dataset.indices[self.train_set.indices[0]]]
        img_name = full_dataset.imgs[dataset.indices[self.train_set.indices[0]]]
        path = self.get_pic_path(self.train_set,0)
        pass

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
        

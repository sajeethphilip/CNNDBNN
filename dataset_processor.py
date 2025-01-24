import os
import json
import zipfile
import tarfile
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torch

class DatasetProcessor:
    def __init__(self, datafile="MNIST", datatype="torchvision", output_dir="data"):
        self.datafile = datafile
        self.datatype = datatype
        self.output_dir = output_dir

    def process(self):
        # First check if input is a directory with train/test structure
        if os.path.isdir(self.datafile):
            train_dir = os.path.join(self.datafile, "train")
            test_dir = os.path.join(self.datafile, "test")
            if os.path.exists(train_dir) and os.path.exists(test_dir):
                # Verify that the folders contain images
                train_has_images = any(f.endswith(('.png', '.jpg', '.jpeg'))
                                     for f in os.listdir(train_dir))
                test_has_images = any(f.endswith(('.png', '.jpg', '.jpeg'))
                                    for f in os.listdir(test_dir))

                if train_has_images and test_has_images:
                    print("Valid dataset folder structure detected. Using existing data.")
                    return train_dir, test_dir

        # If not a valid folder structure, proceed with normal processing
        if self.datatype == "torchvision":
            return self._process_torchvision()
        elif self.datatype == "custom":
            return self._process_custom()
        else:
            raise ValueError("Unsupported datatype. Use 'torchvision' or 'custom'.")


    def _process_torchvision(self):
        dataset_class = getattr(datasets, self.datafile, None)
        if dataset_class is None:
            raise ValueError(f"Torchvision dataset {self.datafile} not found.")

        train_dataset = dataset_class(root=self.output_dir, train=True, download=True)
        test_dataset = dataset_class(root=self.output_dir, train=False, download=True)

        train_dir = os.path.join(self.output_dir, self.datafile, "train")
        test_dir = os.path.join(self.output_dir, self.datafile, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for img, label in train_dataset:
            class_dir = os.path.join(train_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f"{len(os.listdir(class_dir))}.png"))

        for img, label in test_dataset:
            class_dir = os.path.join(test_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f"{len(os.listdir(class_dir))}.png"))

        return train_dir, test_dir

    def _process_custom(self):
        # If it's a directory, check for train/test structure
        if os.path.isdir(self.datafile):
            train_dir = os.path.join(self.datafile, "train")
            test_dir = os.path.join(self.datafile, "test")

            if os.path.exists(train_dir) and os.path.exists(test_dir):
                print(f"Using existing directory structure in {self.datafile}")
                return train_dir, test_dir
            else:
                raise ValueError(f"Directory {self.datafile} must contain 'train' and 'test' subdirectories")

        # If it's a file, process compressed data
        extract_dir = os.path.join(self.output_dir, os.path.splitext(os.path.basename(self.datafile))[0])
        os.makedirs(extract_dir, exist_ok=True)

        if self.datafile.endswith(('.zip')):
            with zipfile.ZipFile(self.datafile, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif self.datafile.endswith(('.tar.gz', '.tgz', '.tar')):
            with tarfile.open(self.datafile, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            raise ValueError("Input must be either a directory with train/test folders or a compressed file (zip/tar.gz)")

        train_dir = os.path.join(extract_dir, "train")
        test_dir = os.path.join(extract_dir, "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise ValueError("Extracted dataset must have 'train' and 'test' folders")

        return train_dir, test_dir


    def generate_json(self, train_dir, test_dir):
        # Find first image and use it as reference
        first_image_path = None
        for root, _, files in os.walk(train_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    first_image_path = os.path.join(root, file)
                    break
            if first_image_path:
                break

        if not first_image_path:
            raise ValueError("No images found in the train directory.")

        # Get image properties from first image only
        with Image.open(first_image_path) as img:
            most_common_size = img.size
            in_channels = 1 if img.mode == "L" else 3

        # Count classes (subdirectories)
        num_classes = len([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])

        # Set standard values for mean and std
        mean = [0.485, 0.456, 0.406] if in_channels == 3 else [0.5]
        std = [0.229, 0.224, 0.225] if in_channels == 3 else [0.5]

        json_data = {
            "dataset": {
                "name": os.path.basename(self.datafile),
                "type": self.datatype,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "input_size": list(most_common_size),
                "mean": mean,
                "std": std,
                "train_dir": train_dir,
                "test_dir": test_dir
            },
            "model": {
                "feature_dims": 128,
                "learning_rate": 0.001
            },
            "training": {
                "batch_size": 32,
                "epochs": 20,
                "num_workers": 4,
                "cnn_training": {
                    "resume": True,
                    "fresh_start": False,
                    "min_loss_threshold": 0.01,
                    "checkpoint_dir": "Model/cnn_checkpoints"
                }
            },
            "execution_flags": {
                "mode": "train_and_predict",
                "use_previous_model": True,
                "fresh_start": False
            }
        }

        json_path = os.path.join(self.output_dir, f"{os.path.basename(self.datafile)}.json")
        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"JSON file created at {json_path}")


def main():
    try:
        datafile = input("Enter dataset name or path (default: MNIST): ").strip() or "MNIST"

        datatype = input("Enter dataset type (torchvision/custom) (default: torchvision): ").strip() or "torchvision"
        if datatype=='torchvision':
            datafile=datafile.upper()
        processor = DatasetProcessor(datafile=datafile, datatype=datatype)
        train_dir, test_dir = processor.process()
        processor.generate_json(train_dir, test_dir)

        print(f"Dataset processed successfully!")
        print(f"Train directory: {train_dir}")
        print(f"Test directory: {test_dir}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

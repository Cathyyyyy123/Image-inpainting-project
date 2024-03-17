import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageInpaintingDataset(Dataset):
    def __init__(self, category_dirs, transform=None):
        """
        Args:
            category_dirs (list of tuples): A list where each tuple contains the paths 
                to the unmasked and masked directories for a specific category.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.file_pairs = []  # This will store tuples of (image_path, mask_path)

        # Iterate over all categories to collect file paths
        for unmasked_dir, masked_dir in category_dirs:
            unmasked_files = [f for f in os.listdir(unmasked_dir) if os.path.isfile(os.path.join(unmasked_dir, f))]
            for file_name in unmasked_files:
                image_path = os.path.join(unmasked_dir, file_name)
                mask_path = os.path.join(masked_dir, file_name)
                self.file_pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.file_pairs[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define transformations, if necessary
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),  # Resize if needed, adjust the size as required
])

# Assuming 'training_data' is the main directory containing 'Cat', 'Dog', etc.
base_dir = 'path/to/training_data'
categories = ['Cat', 'Dog', 'Elephant', 'Tiger']
category_dirs = [(os.path.join(base_dir, cat, 'Unmasked_Train'), os.path.join(base_dir, cat, 'Masked_Train')) for cat in categories]

# Create the dataset
dataset = ImageInpaintingDataset(
    category_dirs=category_dirs,
    transform=transform
)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Example usage
for images, masks in dataloader:
    # Implement training loop here
    pass
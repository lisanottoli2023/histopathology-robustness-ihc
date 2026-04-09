import os
from torch.utils.data import Dataset
from PIL import Image

class IHCDataset(Dataset):
    def __init__(self, root_dir, transform=None, stains_filter=None):
        self.root_dir = root_dir
        self.transform = transform
        self.stains_filter = stains_filter
        self.samples = []
        self.class_to_idx = {}

        classes = sorted(os.listdir(root_dir))

        for class_idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = class_idx
            class_dir = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            for img_name in sorted(os.listdir(class_dir)):

                img_path = os.path.join(class_dir, img_name)

                if not img_name.endswith(".png"):
                    continue

                
                parts = img_name.split("-")
                stain = parts[1]

                
                if stains_filter is not None and stain not in stains_filter:
                    continue

                self.samples.append(
                    (img_path, class_idx, stain)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, stain = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, stain

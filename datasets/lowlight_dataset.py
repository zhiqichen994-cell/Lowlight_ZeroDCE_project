import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LowLightDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image
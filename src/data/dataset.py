import torch
from PIL import Image

class BinarySegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, df, transform=None):
        self.root = root
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        input_path = self.df.iloc[index]["input_path"]
        target_path = self.df.iloc[index]["target_path"]
        input = Image.open(self.root + "/" + input_path)
        target = Image.open(self.root + "/" + target_path)
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        return input_path, target_path, input, target
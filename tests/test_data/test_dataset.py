import os
import sys
import unittest

import torch
import torch.nn as nn
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import BinarySegmentationDataset

class TestUNet(unittest.TestCase):    
    def setUp(self):
        root = "./data/raw/preprocessed"
        path = "./data/csvs//dataset1.csv"
        df = pd.read_csv(path)
        self.dataset = BinarySegmentationDataset(root, df, transform=None)

    def test_len(self):
        self.assertEqual(len(self.dataset), 64)


if __name__ == '__main__':
    unittest.main()
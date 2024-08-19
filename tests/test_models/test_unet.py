import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.unet import UNet


class TestUNet(unittest.TestCase):    
    def setUp(self):
        self.model = UNet()

    def test_forward_shape(self):
        # 入力が与えられたときに、出力の形状が正しいかをテスト
        input_tensor = torch.randn(16, 1, 512, 512)
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, (16, 1, 512, 512))

    def test_parameter_count(self):
        # 学習パラメータ数が、理論値通りかをテスト
        expected_param_count = 7762465
        # 実際に計算した学習パラメータの合計数
        actual_param_count = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(actual_param_count, expected_param_count)


if __name__ == '__main__':
    unittest.main()
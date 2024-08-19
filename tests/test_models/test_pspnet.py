import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.pspnet import PSPNet


class TestPSPNet(unittest.TestCase):    
    def setUp(self):
        self.model = PSPNet(n_classes=21)

    def test_forward_shape(self):
        # 入力が与えられたときに、出力の形状が正しいかをテスト
        input_tensor = torch.randn(2, 3, 475, 475)
        outputs = self.model(input_tensor)
        self.assertEqual(outputs[0].shape, (2, 21, 475, 475))

    def test_parameter_count(self):
        # 学習パラメータ数が、理論値通りかをテスト
        expected_param_count = 49081578
        # 実際に計算した学習パラメータの合計数
        actual_param_count = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(actual_param_count, expected_param_count)


if __name__ == '__main__':
    unittest.main()
import unittest

import numpy as np
import torch

from gabp import GABPBase


class TestGABase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("UnitTest Begin...")
        self.input_size = 21
        self.hidden_number = 5
        self.output_size = 1
        net = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_number),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_number, self.output_size)
        )
        self.ga = GABPBase(net, self.input_size, self.hidden_number, self.output_size, 100, 500, 0.05, 0.7)

    def test_simple_ranking(self):
        test_array_1 = [i for i in range(1, 101)]
        print(test_array_1)
        a = [1.111111111111111111111111111111111111111112, 1.111111111111111111111111111111111111111111]
        a = np.array(a)
        self.assertAlmostEqual(a[0], a[1])
        test_array_2 = []
        for i in range(10):
            step = 1 if i % 2 == 0 else -1
            test_array_2 += [j for j in range(i * 10, (i + 1) * 10)][::step]
        print(test_array_2)

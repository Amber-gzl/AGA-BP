import unittest
import numpy as np
import torch
from ga.gabp import GABPBase


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
        test_array_1 = np.array([i for i in range(1, 101)], dtype=np.float32)
        result_1 = self.ga.simple_ranking(test_array_1)
        self.assertAlmostEqual(result_1[0], 2)
        test_array_2 = []
        for i in range(10):
            step = 1 if i % 2 == 0 else -1
            test_array_2 += [j for j in range(i * 10 + 1, (i + 1) * 10 + 1)][::step]
        test_array_2 = np.array(test_array_2, dtype=np.float32)
        result_2 = self.ga.simple_ranking(test_array_2)
        self.assertAlmostEqual(result_2[0], 2)

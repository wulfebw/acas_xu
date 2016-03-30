import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'reinforcement_learning')))

class TestQNetwork(unittest.TestCase):

    def test_network_initialization(self):
        pass 

if __name__ == '__main__':
    unittest.main()
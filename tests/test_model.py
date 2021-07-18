# Import dependencies and append disease_spread package to path

import unittest
import pathlib
import os
import sys
filePath = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(filePath,".."))
from disease_spread.rvalue_model import run_govukmodel


# Insert test classes
class Test_Model(unittest.TestCase):
    def test_model(self):
        """

        :return:
        """
        finished = False
        x = run_govukmodel(save_output=False)
        finished = True
        self.assertEqual(finished, True)

# write a testcase to test class method age_to_bin of the UTKFaceDataset class in src/dataset.py

import unittest
from dataset import UTKFaceDataset

class TestUTKFaceDataset(unittest.TestCase):
    def test_age_to_bin(self):
        dataset = UTKFaceDataset(csv_path="dummy.csv")  # We won't actually read the file

        # Test age binning
        self.assertEqual(dataset.age_to_bin(5), "0-9")
        self.assertEqual(dataset.age_to_bin(15), "10-19")
        self.assertEqual(dataset.age_to_bin(25), "20-29")
        self.assertEqual(dataset.age_to_bin(35), "30-39")
        self.assertEqual(dataset.age_to_bin(45), "40-49")
        self.assertEqual(dataset.age_to_bin(55), "50-59")
        self.assertEqual(dataset.age_to_bin(65), "60-69")
        self.assertEqual(dataset.age_to_bin(75), "70-79")
        self.assertEqual(dataset.age_to_bin(85), "80+") 


if __name__ == "__main__":    
    unittest.main()  


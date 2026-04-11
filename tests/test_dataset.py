import unittest
from src.dataset import UTKFaceDataset

class TestUTKFaceDataset(unittest.TestCase):
    def test_age_to_bin(self):
        test_cases = [
            (5,  0),  # 0-9
            (15, 1),  # 10-19
            (25, 2),  # 20-29
            (35, 3),  # 30-39
            (45, 4),  # 40-49
            (55, 5),  # 50-59
            (65, 6),  # 60-69
            (75, 7),  # 70-79
            (85, 8),  # 80+
        ]

        for age, expected_bin in test_cases:
            with self.subTest(age=age):
                self.assertEqual(UTKFaceDataset.age_to_bin(age), expected_bin)


if __name__ == "__main__":
    unittest.main()

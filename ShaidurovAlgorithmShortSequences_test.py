import unittest
from ShaidurovAlgorithm import ShaidurovAlgorithm
import numpy as np

class ShaidurovAlgorithmTestCases(unittest.TestCase):
    algorithm = ShaidurovAlgorithm()

    def test_same_sequences(self):
        sequence = np.array(list("AAAA"))
        pattern = np.array(list("AAAA"))

        conclusion = self.algorithm.get_conclusion(sequence, pattern)

        np.testing.assert_allclose(conclusion, [1, 2, 3, 4, 3, 2, 1])

    def test_different_sequences(self):
        sequence = np.array(list("AACCC"))
        pattern = np.array(list("TTGG"))

        conclusion = self.algorithm.get_conclusion(sequence, pattern)

        np.testing.assert_allclose(conclusion, np.zeros(8, dtype=int))

    def test_common_prefix(self):
        sequence = np.array(list("AACCC"))
        pattern = np.array(list("AAC"))

        conclusion = self.algorithm.get_conclusion(sequence, pattern)

        np.testing.assert_allclose(conclusion, np.zeros(8, dtype=int))

if __name__ == '__main__':
    unittest.main()

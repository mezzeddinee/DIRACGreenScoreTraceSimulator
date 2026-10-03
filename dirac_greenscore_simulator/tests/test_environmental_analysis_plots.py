import sys
import unittest
from pathlib import Path

import numpy as np


SIMULATOR_DIR = Path(__file__).resolve().parents[1]
if str(SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_DIR))

from environmental_analysis.plot_multisite_interval_efficiency_over_time import (
    centered_window_ratio,
)


class EnvironmentalAnalysisPlotTests(unittest.TestCase):
    def test_centered_window_ratio_preserves_single_bin_length(self):
        result = centered_window_ratio(
            np.asarray([12.0]),
            np.asarray([3.0]),
            window=3,
        )

        np.testing.assert_allclose(result, np.asarray([4.0]))

    def test_centered_window_ratio_preserves_two_bin_length(self):
        result = centered_window_ratio(
            np.asarray([4.0, 6.0]),
            np.asarray([2.0, 3.0]),
            window=3,
        )

        np.testing.assert_allclose(result, np.asarray([2.0, 2.0]))

    def test_centered_window_ratio_matches_previous_multi_bin_calculation(self):
        numerator = np.asarray([3.0, 6.0, 12.0, 9.0, 15.0])
        denominator = np.asarray([1.0, 2.0, 4.0, 3.0, 5.0])
        kernel = np.ones(3, dtype=float)
        expected = np.convolve(numerator, kernel, mode="same") / np.convolve(
            denominator,
            kernel,
            mode="same",
        )

        result = centered_window_ratio(numerator, denominator, window=3)

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

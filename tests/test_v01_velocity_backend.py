import unittest

import numpy as np

from tools.velocity_waveform_backend import counts_to_velocity_mps


class V01VelocityBackendTests(unittest.TestCase):
    def test_counts_use_channel_sensitivity_once(self):
        counts = np.array([-20, 0, 40], dtype=np.int32)
        actual = counts_to_velocity_mps(counts, 20.0, "m/s")
        np.testing.assert_allclose(actual, [-1.0, 0.0, 2.0])

    def test_rejects_non_velocity_units(self):
        with self.assertRaisesRegex(ValueError, "Expected CH velocity unit"):
            counts_to_velocity_mps(np.array([1]), 2.0, "nm/s")

    def test_rejects_invalid_sensitivity(self):
        with self.assertRaisesRegex(ValueError, "Invalid counts_per_physical_unit"):
            counts_to_velocity_mps(np.array([1]), 0.0, "m/s")


if __name__ == "__main__":
    unittest.main()

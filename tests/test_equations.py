import unittest
import numpy as np

from orochi.pixia.equation.sac import sac, solid_angle_correction
from orochi.pixia.equation.polarization import polarization, polar_correction
from orochi.pixia.equation.detector_absorption import detector_absorption, absorption_correction
from orochi.pixia.equation.air_absorption import air_absorption, air_absorption_correction


class TestPixiaEquations(unittest.TestCase):
    def setUp(self):
        self.t0 = np.array([0.0, 0.0, 0.0])
        # Define 3 pixel positions in 3D: (x, y, z)
        self.coords = np.array([
            [0.0, 0.0, 227.6],
            [50.0, 50.0, 227.6],
            [100.0, 100.0, 227.6]
        ])
        self.normals = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0]
        ])
        self.par = {
            "distance": 227.6,
            "polar": 0.99,
            "rotation": 0.0,
            "monochro_direction": 0.0,
            "monochro_angle": 0.0,
            "monochro_mosaic": 1.0,
            "thickness": 0.05,
            "mur": 1.2,
            "air_type": 1,
            "energy": 68.48,
        }

    def test_sac(self):
        val = sac(self.t0, self.coords, self.par)
        self.assertEqual(len(val), 3)
        self.assertAlmostEqual(val[0], 1.0, places=4)
        self.assertTrue(np.all(val >= 1.0))

    def test_polarization(self):
        tt = np.array([0.1, 0.3, 0.5])
        beta = np.array([0.0, np.pi / 4, np.pi / 2])
        val = polarization(tt, beta, self.par)
        self.assertEqual(len(val), 3)
        self.assertTrue(np.all(val > 0.0))

    def test_detector_absorption(self):
        val = detector_absorption(self.t0, self.coords, self.normals, self.par)
        self.assertEqual(len(val), 3)
        self.assertTrue(np.all(val > 0.0))

    def test_air_absorption(self):
        val = air_absorption(self.t0, self.coords, self.par)
        self.assertEqual(len(val), 3)
        self.assertTrue(np.all(val > 0.0))


if __name__ == "__main__":
    unittest.main()


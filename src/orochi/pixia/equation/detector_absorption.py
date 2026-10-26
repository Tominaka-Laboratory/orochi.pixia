"""Sensor absorption correction assuming thick detector sensors."""

import numpy as np

# t0: sample position, (x, y, z) coordinate
# coordinates: coordinates of pixels (x, y, z)
# normals: normals of pixels
# par: PIXpar / Pixia instance


def absorption_correction(t0, coordinates, normals, par):
    """Computes oblique transmission and quantum efficiency through the detector sensor."""
    if par["thickness"] == 0.0 or par["mur"] == 0.0:
        return np.ones(coordinates.shape[0])
    else:
        center = coordinates - t0

        inners = np.sum(center * normals, axis=-1)
        n = np.linalg.norm(center, axis=-1) * np.linalg.norm(normals, axis=-1)
        angles = np.arccos(np.clip(inners / n, -1.0, 1.0))

        path = par["thickness"] / np.cos(angles)

        # approximation as a trapezoid
        detector_absorption = 1.0 - np.exp(-1.0 * par["mur"] * path)
        detector_absorption /= np.amax(detector_absorption)

        return detector_absorption


# Aliases
detector_absorption = absorption_correction
absorption = absorption_correction


"""Polarization correction for 2D pixel detectors.

References
----------
Fundamentals of Crystallography, 3rd ed., pp. 384-.
"""

import numpy as np

# tt: twotheta in radians
# beta: beta in radians
# par: PIXpar / Pixia instance


def polar_correction(tt, beta, par):
    """Computes polarization correction factor for synchrotron or laboratory sources."""
    if par["polar"] is None:
        rho = beta - np.radians(par["monochro_direction"])
    else:
        rho = beta - np.radians(par["rotation"])

    sqcostt = np.square(np.cos(tt))
    sqsintt = np.square(np.sin(tt))
    sqcosrho = np.square(np.cos(rho))
    sqsinrho = np.square(np.sin(rho))
    cos2rho = np.cos(2.0 * rho)

    if par["polar"] is None:  # nonpolar / lab source with crystal monochromator
        monochro = np.abs(
            np.cos(np.radians(par["monochro_angle"])) ** par["monochro_mosaic"]
        )
        polarization = (
            (sqcostt * sqcosrho + sqsinrho) * monochro
            + (sqcostt * sqsinrho + sqcosrho)
        ) / (1.0 + monochro)
    else:  # synchrotron radiation
        polarization = (1.0 + sqcostt - par["polar"] * sqsintt * cos2rho) / 2.0

    return polarization


# Aliases
polarization = polar_correction
polarization_correction = polar_correction


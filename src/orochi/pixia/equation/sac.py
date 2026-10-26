"""Solid Angle Correction (SAC) and oblique incidence model for 2D pixel detectors."""

import numpy as np

# t0: sample position, (x, y, z) coordinate
# coordinates: coordinates of pixels (x, y, z)
# par: PIXpar / Pixia instance


def solid_angle_correction(t0, coordinates, par):
    """Computes exact solid angle / oblique incidence correction factor for pixel detectors.

    For a planar pixel detector orthogonal to the beam axis:
        Omega = A * cos(alpha) / rho^2 = A * L0 / rho^3
        sac = Omega_0 / Omega = (rho / L0)^3 = 1 / cos^3(2theta)
    Multiplying measured intensity by `sac` yields the angle-independent intensity per unit solid angle.
    """
    beam = coordinates - t0
    beam_length = np.linalg.norm(beam, axis=-1)
    beam_height = float(par["distance"])

    # Normal distance along detector axis
    if beam.ndim > 1:
        cos_alpha = np.clip(np.abs(beam[..., 2]) / beam_length, 1e-6, 1.0)
    else:
        cos_alpha = np.clip(np.abs(beam[2]) / beam_length, 1e-6, 1.0)

    sac_factor = (beam_length / beam_height) ** 2 / cos_alpha
    return sac_factor


# Aliases for concise and modular reference
sac = solid_angle_correction


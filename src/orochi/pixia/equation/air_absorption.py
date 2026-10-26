"""Air path absorption correction for 2D pixel detectors."""

import numpy as np
from orochi.pixia.database.air_absorption_seltzer import air_absorption_function

# t0: sample position, (x, y, z) coordinate
# coordinates: coordinates of pixels (x, y, z)
# par: PIXpar / Pixia instance


def air_absorption_correction(t0, coordinates, par):
    """Computes X-ray attenuation along the sample-to-detector air path."""
    if par.get("air_type") is None:
        return np.ones(coordinates.shape[0])

    air_density = 1.293 * 10**-3  # g/cm3

    functions = air_absorption_function()
    energy_val = float(par["energy"])
    if callable(functions):
        # Database aa(energy) expects eV (it divides by 1000 to get keV)
        energy_input = energy_val * 1000.0 if energy_val < 1000.0 else energy_val
        coeffs = functions(energy_input)
        air_idx = int(par.get("air_type", 0))
        # Ensure air_idx is in range [0, len(coeffs)-1]
        air_idx = min(max(air_idx, 0), len(coeffs) - 1)
        mu = float(coeffs[air_idx])
    else:
        mu = float(functions[int(par["air_type"])](energy_val))

    distance = np.linalg.norm(coordinates - t0, axis=-1)

    rho_air = (air_density * mu) / 10.0  # rho in mm-1

    air_abs = np.exp(-1.0 * rho_air * distance)
    max_abs = np.amax(air_abs)
    if max_abs > 0:
        air_abs /= max_abs

    return air_abs


# Aliases
air_absorption = air_absorption_correction
air = air_absorption_correction


import os
import json
import numpy as np
from matplotlib import pyplot as plt

from orochi.pixia.pixia import Pixia, calculator as pix
from orochi.pixia.calibrator import Calibrator


def typical_calibration(sample_file, background_file, mask_file, pixia_par, out_path):
    """Calibrates 2D detector geometry against a standard powder diffraction pattern (e.g., CeO2)."""
    os.makedirs(out_path, exist_ok=True)
    file_name = os.path.basename(sample_file).split(".")[0]

    before_params = {
        k: float(pixia_par[k])
        for k in ["rambda", "distance", "x0", "y0", "tiltx", "tilty"]
    }

    x1, y1, x2, y2, pixia_par = two_dimensional_detector(
        sample_file, background_file, mask_file, "ceo2", pixia_par
    )

    after_params = {
        k: float(pixia_par[k])
        for k in ["rambda", "distance", "x0", "y0", "tiltx", "tilty"]
    }

    with open(os.path.join(out_path, f"pixia_{file_name}_calibration.json"), "w") as f:
        json.dump({"before": before_params, "after": after_params}, f, indent=2)

    print("\nbefore -> after")
    for key in ["rambda", "distance", "x0", "y0", "tiltx", "tilty"]:
        print(f"{key}: {before_params[key]:.6f} -> {after_params[key]:.6f}")

    data = np.column_stack([x2, y2])
    np.savetxt(os.path.join(out_path, f"pixia_cal_tt1d_i1d_{file_name}.txt"), data)

    asave_name = os.path.join(out_path, f"pixia_{file_name}_after_optimization")
    plt.figure()
    plt.plot(x1, y1, label="before optimization")
    plt.plot(x2, y2, label="after optimization")
    plt.legend()
    plt.savefig(asave_name + ".png")
    plt.close()


def two_dimensional_detector(sample_file, background_file, mask_file, standard_type, pixia_par):
    """Runs Optuna multi-stage geometry calibration for a 2D area detector."""
    pixia_cal = pix(pixia_par)
    pixia_cal["sample_file"] = sample_file
    pixia_cal["background_file"] = background_file
    pixia_cal["mask_file"] = mask_file
    pixia_cal.run()

    x1 = pixia_cal.tt1d
    y1 = pixia_cal.i1d

    pixia_calibrator = Calibrator(pixia=pixia_cal, standard=standard_type)
    pixia_calibrator.set_mask()

    pixia_calibrator.optimize(
        n_trials=[80, 80, 20, 20],
        scales=[1.0, 1.0, 0.2, 0.2],
        targets=["center", "peak_angle", "center", "peak_angle"],
        key_diff={
            "rambda": 0.01,
            "x0": 1.0,
            "y0": 1.0,
            "tiltx": 0.1,
            "tilty": 0.1,
            "distance": 1.0,
        },
    )

    pixia_calibrator.reset_mask()
    pixia_cal.convert_selected()
    pixia_cal.twotheta_intensity()

    x2 = pixia_cal.tt1d
    y2 = pixia_cal.i1d

    return x1, y1, x2, y2, pixia_cal
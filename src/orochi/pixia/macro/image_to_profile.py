import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from orochi.pixia.pixia import Pixia, calculator


def conversion(sample_file, background_file, mask_file, pixia_par):
    """Integrates a 2D image with background and mask into a 1D profile and dummy baseline."""
    pixia_cal = calculator(pixia_par)
    pixia_cal["sample_file"] = sample_file
    pixia_cal["background_file"] = background_file
    pixia_cal["mask_file"] = mask_file

    pixia_cal.run()

    x = pixia_cal.tt1d
    y = pixia_cal.i1d
    yb = pixia_cal.get_background_curve()

    return x, y, yb


def typical_conversion(sample_file, background_file, mask_file, pixia_par, out_path):
    """Runs 2D-to-1D integration and saves both the text data and overview PNG/pickle figure."""
    os.makedirs(out_path, exist_ok=True)
    if isinstance(sample_file, str):
        file_name = os.path.basename(sample_file).split(".")[0]
    else:
        file_name = os.path.basename(sample_file[0]).split(".")[0]

    x, y, yb = conversion(sample_file, background_file, mask_file, pixia_par)

    data = np.column_stack([x, y, yb])
    np.savetxt(f"{out_path}/{file_name}_pixia_data.txt", data, fmt="%.5e")

    fig = plt.figure()
    plt.plot(x, y, label="profile")
    plt.legend()
    name = f"{out_path}/{file_name}_pixia"
    plt.savefig(name + ".png")
    save_pkl(fig, name)
    plt.close()


def simple_conversion(pixia_par, out_path):
    """Runs 2D-to-1D integration using file paths already configured in `pixia_par`."""
    os.makedirs(out_path, exist_ok=True)
    sample_file = pixia_par["sample_file"]
    if isinstance(sample_file, str):
        file_name = os.path.basename(sample_file).split(".")[0]
    else:
        file_name = os.path.basename(sample_file[0]).split(".")[0]

    pixia_cal = calculator(pixia_par)
    pixia_cal.run()

    x = pixia_cal.tt1d
    y = pixia_cal.i1d
    yb = pixia_cal.get_background_curve()

    data = np.column_stack([x, y, yb])
    np.savetxt(f"{out_path}/{file_name}_pixia_data.txt", data, fmt="%.5e")

    fig = plt.figure()
    plt.plot(x, y, label="profile")
    plt.legend()
    name = f"{out_path}/{file_name}_pixia"
    plt.savefig(name + ".png")
    save_pkl(fig, name)
    plt.close()


def save_pkl(fig, file_name):
    with open(file_name + "_fig.pkl", "wb") as ofs:
        pickle.dump(fig, ofs)
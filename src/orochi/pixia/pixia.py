import glob
import os
import json
import numpy as np
from scipy import interpolate as intp
from matplotlib import path as mpltPath

from orochi.base.core import OrochiBase
from .params import default
from .calculator.profile import profile_calculation
from .calculator.rmse import rmse_calculation


class Pixia(OrochiBase):
    """2D X-ray diffraction image integration and geometry/intensity correction module.

    Converts 2D detector images (TIFF, FPD, Imaging Plate) into 1D powder
    diffraction profiles I(2theta) and synthetic uniform-intensity dummy
    baselines for downstream PDF extraction in `orochi.pdfget`.
    """

    def __init__(self):
        self._par = default.par
        self._par_key_info = default.par_key_info
        super().__init__()

        self._name = "orochi.pixia"
        self._id = "0"
        self.digit = np.float64

        self.sample = 1
        self.sample_offset = 0
        self.background = 0
        self.background_offset = 0
        self.gain = 1
        self.mask_image = 1
        self.mask_spots = 1
        self.mask = None
        self.ibgs = None
        self.tt1d = None
        self.i1d = None
        self.d1d = None

        self.input = {}
        self.filetype = {
            "tif": [0, 0],
            "tiff": [0, 0],
            "jpeg": [0, 0],
            "jpg": [0, 0],
            "img": [1, 1],
            "stl": [1, 1],
            "osc": [1, 4],
            "his": [2, 50],
        }

        rambda_val = self["rambda"]
        energy_val = self["energy"]
        if (rambda_val is None or np.isnan(rambda_val)) and (
            energy_val is not None and not np.isnan(energy_val)
        ):
            self.set_rambda_from_energy()
        elif (rambda_val is not None and not np.isnan(rambda_val)) and (
            energy_val is None or np.isnan(energy_val)
        ):
            self.set_energy_from_rambda()

        self._set_equations()

    def __setitem__(self, key, value):
        res = super().__setitem__(key, value)
        base_key = key[0] if isinstance(key, (list, tuple)) else key
        if base_key == "rambda" and self["rambda"] is not None:
            self.set_energy_from_rambda()
        elif base_key == "energy" and self["energy"] is not None:
            self.set_rambda_from_energy()
        return res

    @property
    def tt(self):
        """Scattering angle 2theta grid (degrees) for the integrated 1D profile."""
        return self.tt1d

    @property
    def i(self):
        """Integrated 1D intensity profile."""
        return self.i1d

    @property
    def dummy_baseline(self):
        """Synthetic uniform-intensity dummy baseline profile for normalization."""
        return self.d1d

    def __call__(self, force: bool = False, dummy: bool = True):
        """Executes constraints and runs the 2D-to-1D integration pipeline.

        Parameters
        ----------
        force : bool, optional
            If True, resets internal state to 'INIT' to force full re-evaluation
            without skipping cached steps, by default False.
        dummy : bool, optional
            Whether to compute the synthetic uniform-intensity dummy baseline
            profile for downstream normalization in PDFget. Default is True.
        """
        if force:
            self._state = "INIT"
        super().__call__(force=force)

        self._logger.info("pixia call start...")
        self._set_equations()
        self.read(force_reload=force)
        self.bg_subtraction()

        self._logger.info("selected-region conversion starts...")
        self.convert_selected()
        self.twotheta_intensity()
        if dummy:
            self.get_background_curve()
        self._state = "DONE"
        return self

    def run(self, force: bool = False, dummy: bool = True):
        """Explicit execution entry point for user scripts and interactive environments.

        Parameters
        ----------
        force : bool, optional
            If True, forces full recalculation by bypassing cached state, by default False.
        dummy : bool, optional
            Whether to compute the synthetic uniform-intensity dummy baseline
            profile for downstream normalization in PDFget. Default is True.
        """
        return self.__call__(force=force, dummy=dummy)

    def read_file(self, file_path, target="sample"):
        """Read an image file into the specified target slot.

        Parameters
        ----------
        file_path : str
            Path to the image file (TIFF, HIS, etc.).
        target : str, optional
            Target image slot: "sample", "background", "mask", "sample_offset",
            "background_offset", or "gain". Default is "sample".
        """
        key_map = {
            "sample": "sample_file",
            "background": "background_file",
            "mask": "mask_file",
            "sample_offset": "sample_offset_file",
            "background_offset": "background_offset_file",
            "gain": "gain_file",
        }
        param_key = key_map.get(target, "sample_file")
        self[param_key] = file_path
        if self["sample_file"] is not None:
            self.read()
            self.bg_subtraction()

    def to_json(self, file_path=None):
        """Export integrated 1D profile and module parameters to a JSON dictionary/file.

        Parameters
        ----------
        file_path : str, optional
            If specified, saves the JSON to this path.

        Returns
        -------
        dict
            Dictionary containing 1D curves ('tt', 'i', 'd') and a nested 'params'
            dictionary containing all module parameters.
        """
        json_data = {}
        if self.tt1d is not None and len(self.tt1d) > 0:
            json_data["tt"] = np.array(self.tt1d).tolist()
        if self.i1d is not None and len(self.i1d) > 0:
            json_data["i"] = np.array(self.i1d).tolist()
        if self.d1d is not None and len(self.d1d) > 0:
            json_data["d"] = np.array(self.d1d).tolist()

        from orochi.base.core import NumpyEncoder
        params = {}
        for k in self.par:
            try:
                val = self[k]
                if isinstance(val, (np.bool_, bool)):
                    params[k] = bool(val)
                elif isinstance(val, np.ndarray):
                    params[k] = val.tolist()
                elif isinstance(val, (np.floating, float)):
                    params[k] = None if np.isnan(val) else float(val)
                elif isinstance(val, (np.integer, int)):
                    params[k] = int(val)
                elif isinstance(val, str) or val is None:
                    params[k] = val
                else:
                    params[k] = val
            except Exception:
                pass
        json_data["params"] = params

        if file_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            self._logger.info(f"Saved JSON data to {file_path}")

        return json_data

    def from_json(self, source):
        """Load module parameters and 1D profile from a JSON dictionary or file.

        Parameters
        ----------
        source : str or dict
            File path or pre-loaded dictionary containing 'params' and 1D profile keys.
        """
        if isinstance(source, str):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(source, dict):
            data = source
        else:
            raise TypeError("source must be a file path string or a dict")

        if "params" in data and isinstance(data["params"], dict):
            self.update(data["params"])

        if "tt" in data and data["tt"] is not None:
            self.tt1d = np.array(data["tt"])
        if "i" in data and data["i"] is not None:
            self.i1d = np.array(data["i"])
        if "d" in data and data["d"] is not None:
            self.d1d = np.array(data["d"])

    def save_json(self, file_path):
        """Alias for to_json(file_path)."""
        return self.to_json(file_path)

    def load_json(self, file_path):
        """Alias for from_json(file_path)."""
        return self.from_json(file_path)

    def save_file(self, file_path):
        """Save the integrated 1D profile to a text file or JSON (.json).

        Parameters
        ----------
        file_path : str
            Target file path for output file. If ends with .json, exports JSON format.

        Raises
        ------
        RuntimeError
            If 1D integration has not been performed yet.
        """
        if file_path.endswith(".json"):
            return self.to_json(file_path)

        if self.tt1d is None or self.i1d is None:
            raise RuntimeError("No 1D profile available to save. Run Pixia first.")
        if not os.path.splitext(file_path)[1]:
            file_path = file_path + "_pixia_data.txt"
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        if self.d1d is None:
            self.get_background_curve()
        data = np.column_stack([self.tt1d, self.i1d, self.d1d])
        header = "2theta(deg) intensity dummy_baseline"
        np.savetxt(file_path, data, fmt="%.5e", header=header)
        self._logger.info(f"Saved 1D profile to {file_path}")

    def read(self, force_reload=False):
        """Load sample, background, gain, and mask images and build combined mask."""
        current_files = (
            str(self["sample_file"]),
            str(self["sample_offset_file"]),
            str(self["background_file"]),
            str(self["background_offset_file"]),
            str(self["gain_file"]),
            str(self["mask_file"]),
        )
        if (
            force_reload
            or not hasattr(self, "_cached_files")
            or self._cached_files != current_files
            or getattr(self, "sample", None) is None
        ):
            self._raw_sample = self.read_images(self["sample_file"])
            self._raw_sample_offset = self.read_images(self["sample_offset_file"])
            self._raw_background = self.read_images(self["background_file"])
            self._raw_background_offset = self.read_images(self["background_offset_file"])
            self._raw_gain = self.read_images(self["gain_file"], default=1.0)

            mask_img = self.read_images(self["mask_file"], default=1.0)
            if np.amax(mask_img) != 0:
                mask_img = mask_img / np.amax(mask_img)
            self._raw_mask_image = mask_img

            self.sample = self._raw_sample - self._raw_sample_offset
            self.background = self._raw_background - self._raw_background_offset
            self.gain = self._raw_gain
            self.mask_image = self._raw_mask_image
            self._cached_files = current_files

        self.mask = self.get_mask()


    def read_a_file(self, file):
        """Reads a single 2D detector image file based on its file extension."""

        def read(reader):
            def flat_panel(file):
                data = np.fromfile(file, dtype=np.uint16())
                self["pixy"] = int(data[8])
                self["pixx"] = int(data[9])
                stored_bytes = data[16]

                if stored_bytes == 32:
                    data = np.fromfile(file, dtype=np.uint32())
                    idx = np.arange(
                        int(self["fpd_header"] / 2),
                        self["pixx"] * self["pixy"] + int(self["fpd_header"] / 2),
                        1,
                    )
                    intensity = data[idx].reshape(self["pixy"], self["pixx"]).astype(float)
                else:
                    idx = np.arange(
                        self["fpd_header"],
                        self["pixx"] * self["pixy"] + self["fpd_header"],
                        1,
                    )
                    intensity = data[idx].reshape(self["pixy"], self["pixx"]).astype(float)

                intensity = intensity[::-1, :]
                return intensity

            def tiff(file):
                import tifffile as tif

                intensity = tif.imread(file).astype(self.digit)[::1, :]
                return intensity

            def imaging_plate(file):
                data = np.fromfile(file, dtype=np.uint8())
                v1 = (
                    data[np.arange(0, self["pixx"] * self["pixy"] * 2, 2)]
                    .reshape(self["pixy"], self["pixx"])
                    .astype(float)
                )
                v2 = (
                    data[np.arange(1, self["pixx"] * self["pixy"] * 2, 2)]
                    .reshape(self["pixy"], self["pixx"])
                    .astype(float)
                )

                r1 = np.where(v1 >= 128)
                v1[r1] = v1[r1] - 128

                intensity = v1 * 256 + v2
                intensity[r1] = (intensity[r1] - self["ip_noise"]) * self["ip_hvlv"]
                return intensity

            if reader == 0:
                return tiff(file)
            elif reader == 1:
                return imaging_plate(file)
            elif reader == 2:
                return flat_panel(file)

        ext = os.path.splitext(file)[-1].strip(".").lower()
        reader = self.filetype.get(ext)

        if reader is None:
            try:
                import tifffile as tif
                intensity = tif.imread(file).astype(self.digit)[::1, :]
                return intensity, True
            except Exception:
                return 0, False
        else:
            return read(reader[0]), True

    def read_images(self, files, default=0.0):
        """Reads and averages one or more image files."""
        if isinstance(files, (list, np.ndarray)):
            if np.size(files) == 0 or (np.size(files) == 1 and files[0] is None):
                return default
        else:
            if files is None or files == "":
                return default
            else:
                files = [files]

        data, total = 0.0, 0
        for file in files:
            if file is None or file == "":
                continue
            intensity, ok = self.read_a_file(file)
            if ok:
                total += 1
                data += (intensity - data) / total
                self._logger.info(f"{os.path.basename(file)} was read.")
        if total == 0:
            return default
        return np.array(data, dtype=self.digit)

    def get_mask(self):
        """Constructs the composite 2D pixel mask (1 = valid pixel, 0 = masked pixel)."""

        def edge():
            if None in [self["edge_y1"], self["edge_y2"], self["edge_x1"], self["edge_x2"]]:
                return np.ones((self["pixy"], self["pixx"]))
            else:
                edge_arr = np.zeros((self["pixy"], self["pixx"]))
                edge_arr[
                    self["edge_y1"] : self["edge_y2"],
                    self["edge_x1"] : self["edge_x2"],
                ] = 1.0
                return edge_arr

        def grid():
            mx = self["grid_xw"]
            my = self["grid_yw"]
            gx = self["grid_xp"]
            gy = self["grid_yp"]

            if None in [mx, my, gx, gy]:
                return np.ones((self["pixy"], self["pixx"]))
            else:
                mask = np.zeros((self["pixy"], self["pixx"]))
                if gx == 0 and gy == 0:
                    mask += 1.0
                else:
                    gridx = np.linspace(0, self["pixx"], gx + 1, dtype=int)
                    gridy = np.linspace(0, self["pixy"], gy + 1, dtype=int)
                    for x in range(0, gx):
                        for y in range(0, gy):
                            mask[
                                gridy[y] + my : gridy[y + 1] - my,
                                gridx[x] + mx : gridx[x + 1] - mx,
                            ] = 1.0
                return mask

        def beamstop():
            if self["beamstop_circle"] == 0:
                return np.ones((self["pixy"], self["pixx"]))
            else:
                y, x = np.mgrid[0 : self["pixy"], 0 : self["pixx"]]
                xy_coordinates = np.array([x.flatten(), y.flatten()], dtype=self.digit).T
                xy_coordinates = xy_coordinates.reshape(self["pixy"], self["pixx"], 2) + [0.5, 0.5]
                xy_coordinates -= np.array([self["x0"], self["y0"]])

                radius = np.linalg.norm(xy_coordinates, axis=-1)
                mask = np.where(radius < self["beamstop_circle"], 0, 1)
                return mask

        def beamstop_frame():
            if self["beamstop_frame"] == 0:
                return np.ones((self["pixy"], self["pixx"]))
            else:
                return belt(self["beamstop_frame"], self["beamstop_angle"], positive=False)

        def rectangle():
            rect_list = self.par["mask_rectangle"]["value"]
            if len(rect_list) == 0:
                return np.ones((self["pixy"], self["pixx"]))
            else:
                mask = np.ones((self["pixy"], self["pixx"]))
                for m in np.array(rect_list).reshape(-1, 4):
                    mask[int(m[0]) : int(m[1]), int(m[2]) : int(m[3])] = 0
                return mask

        def polygon():
            poly_list = self.par["mask_polygon"]["value"]
            if len(poly_list) == 0:
                return np.ones((self["pixy"], self["pixx"]))
            else:
                x, y = np.meshgrid(np.arange(self["pixx"]), np.arange(self["pixy"]))
                x, y = x.flatten(), y.flatten()
                xy_coordinates = np.vstack((x, y)).T
                mask = np.ones((self["pixy"], self["pixx"]))

                for pg in poly_list:
                    path = mpltPath.Path(pg)
                    inside = path.contains_points(xy_coordinates, radius=3)
                    grid_mask = inside.reshape((self["pixy"], self["pixx"]))
                    mask[grid_mask] = 0

                return mask

        def window():
            if self["window_width"] == 0:
                return np.ones((self["pixy"], self["pixx"]))
            else:
                return belt(self["window_width"] / 2.0, self["window_angle"], positive=True)

        def belt(window_width, window_angle, positive=True):
            zero_exception = 0.00001
            limit = 1

            y, x = np.mgrid[0 : self["pixy"], 0 : self["pixx"]]
            xy_coordinates = np.array([x.flatten(), y.flatten()], dtype=float).T
            centers = xy_coordinates.reshape(self["pixy"], self["pixx"], 2) + [0.5, 0.5]
            xy = centers - np.array([self["x0"], self["y0"]])

            if window_angle == 90.0:
                p_right = np.array([self["x0"], self["y0"]]) + [window_width, 0.0]
                p_right_direction = np.array([self["x0"], self["pixy"] + 0.5]) + [window_width, 0.0]
                p_left = np.array([self["x0"], self["y0"]]) - [window_width, 0.0]
                p_left_direction = np.array([self["x0"], self["pixy"] + 0.5]) - [window_width, 0.0]
                p_direction = np.array([self["x0"], self["pixy"] + 0.5])

            elif window_angle == -90.0:
                p_right = np.array([self["x0"], self["y0"]]) + [window_width, 0.0]
                p_right_direction = np.array([self["x0"], 0.5]) + [window_width, 0.0]
                p_left = np.array([self["x0"], self["y0"]]) - [window_width, 0.0]
                p_left_direction = np.array([self["x0"], 0.5]) - [window_width, 0.0]
                p_direction = np.array([self["x0"], 0.5])

            elif window_angle in [0.0, -0.0]:
                p_right = np.array([self["x0"], self["y0"]]) - [0.0, window_width]
                p_right_direction = np.array([self["pixx"] + 0.5, self["y0"]]) - [0.0, window_width]
                p_left = np.array([self["x0"], self["y0"]]) + [0.0, window_width]
                p_left_direction = np.array([self["pixx"] + 0.5, self["y0"]]) + [0.0, window_width]
                p_direction = np.array([self["pixx"] + 0.5, self["y0"]])

            elif window_angle in [-180.0, 180.0]:
                p_right = np.array([self["x0"], self["y0"]]) - [0.0, window_width]
                p_right_direction = np.array([0.5, self["y0"]]) - [0.0, window_width]
                p_left = np.array([self["x0"], self["y0"]]) + [0.0, window_width]
                p_left_direction = np.array([0.5, self["y0"]]) + [0.0, window_width]
                p_direction = np.array([0.5, self["y0"]])

            else:
                pn = 1.0 if (-90.0 < window_angle < 90.0) else -1.0

                d_tan = np.tan(np.radians(window_angle))
                n_tan_right = np.tan(np.radians(window_angle - 90.0))
                n_tan_left = np.tan(np.radians(window_angle + 90.0))

                y = xy[:, :, 1]
                x = xy[:, :, 0]
                x[x == 0.0] = zero_exception
                radius = np.linalg.norm(xy, axis=-1)
                radius[radius == 0.0] = zero_exception

                right = np.where((x >= 0) & (np.abs(radius - window_width) < 2))
                p_right = np.argmin(np.abs(y[right] - n_tan_right * x[right]))
                p_right = centers[right][p_right]

                left = np.where((x < 0) & (np.abs(radius - window_width) < 2))
                p_left = np.argmin(np.abs(y[left] - n_tan_left * x[left]))
                p_left = centers[left][p_left]

                direction = np.where((x * pn >= 0) & (np.abs(y - d_tan * x) < limit))
                p_direction = np.argmin(1.0 / radius[direction])
                p_direction = centers[direction][p_direction]

                xy_right = centers - p_right
                y = xy_right[:, :, 1]
                x = xy_right[:, :, 0]
                x[x == 0.0] = zero_exception

                direction = np.where((x * pn >= 0) & (np.abs(y - d_tan * x) < limit))
                radius = np.linalg.norm(xy_right[direction], axis=-1)
                radius[radius == 0.0] = zero_exception

                p_right_direction = np.argmin(1.0 / radius)
                p_right_direction = centers[direction][p_right_direction]

                xy_left = centers - p_left
                y = xy_left[:, :, 1]
                x = xy_left[:, :, 0]
                x[x == 0.0] = zero_exception

                direction = np.where((x * pn >= 0) & (np.abs(y - d_tan * x) < limit))
                radius = np.linalg.norm(xy_left[direction], axis=-1)
                radius[radius == 0.0] = zero_exception

                p_left_direction = np.argmin(1.0 / radius)
                p_left_direction = centers[direction][p_left_direction]

            pg = (
                np.array([p_left, p_right, p_right_direction, p_direction, p_left_direction])
                .reshape((-1, 2))
                .tolist()
            )
            path = mpltPath.Path(pg)
            inside = path.contains_points(xy_coordinates, radius=3)
            grid_mask = inside.reshape((self["pixy"], self["pixx"]))

            if positive:
                mask = np.zeros((self["pixy"], self["pixx"]))
                mask[grid_mask] = 1
            else:
                mask = np.ones((self["pixy"], self["pixx"]))
                mask[grid_mask] = 0
            return mask

        mask = (
            beamstop()
            * beamstop_frame()
            * edge()
            * grid()
            * rectangle()
            * polygon()
            * window()
            * self.mask_image
            * self.mask_spots
        )
        mask = np.round(np.array(mask, dtype=float))
        return mask

    def bg_subtraction(self):
        """Applies deadtime correction, background subtraction, and flat-field gain correction."""
        self._logger.debug("bg_subtraction start...")
        si = self.deadtime_correction(self.sample)
        if not isinstance(self.background, (int, float)):
            bi = self.deadtime_correction(self.background) * self["background_scale"]
        else:
            bi = 0.0

        if not isinstance(self.gain, (int, float)):
            intensity = (si - bi) / self.gain
        else:
            intensity = si - bi

        self.ibgs = intensity

    def deadtime_correction(self, image):
        """Applies non-paralyzable detector deadtime correction."""
        if self["deadtime"] == 0:
            return image
        else:
            i0 = 10 ** np.linspace(1.0, 7.0, 100)
            iobs = i0 / (1.0 + i0 * self["deadtime"] / 10.0**6)
            corrected = intp.interp1d(iobs, i0, fill_value="extrapolate")(image)
            return corrected

    def remove_spots(self, image):
        """Detects and masks sharp single-crystal or cosmic-ray spots."""
        r1 = np.where(image > 0, 1, 0)
        image = np.array(image, copy=True)
        image[image <= 0] = 1.0

        diff_y = np.diff(image, axis=0) / image[1:, :]
        diff_y = np.append(diff_y, np.zeros(image.shape[0])[np.newaxis, :], axis=0)
        diff_x = np.diff(image, axis=1) / image[:, 1:]
        diff_x = np.append(diff_x, np.zeros(image.shape[1])[:, np.newaxis], axis=1)

        self.t0 = (
            np.array(
                [np.sin(np.radians(self["tiltx"])), np.sin(np.radians(self["tilty"])), 1.0],
                dtype=self.digit,
            )
            * self["distance"]
        )
        xyz_coordinates, _ = self.get_coordinates()

        xyz_coord = xyz_coordinates.reshape(-1, 3)
        tt = self.get_tt_radian(xyz_coord).reshape(self["pixy"], self["pixx"])
        beta = self.get_beta_radian(xyz_coord).reshape(self["pixy"], self["pixx"])

        diff_tt_y = np.diff(tt, axis=0)
        diff_tt_y = np.append(diff_tt_y, np.zeros(image.shape[0])[np.newaxis, :], axis=0)
        diff_tt_x = np.diff(tt, axis=1)
        diff_tt_x = np.append(diff_tt_x, np.zeros(image.shape[1])[:, np.newaxis], axis=1)

        diff_be_y = np.diff(beta, axis=0)
        diff_be_y = np.append(diff_be_y, np.zeros(image.shape[0])[np.newaxis, :], axis=0)
        diff_be_x = np.diff(beta, axis=1)
        diff_be_x = np.append(diff_be_x, np.zeros(image.shape[1])[:, np.newaxis], axis=1)

        r2 = np.where(
            (np.abs(diff_y) > self["outlier"]) & (np.abs(diff_tt_y) < 5 * np.abs(diff_be_y)),
            1,
            0,
        )
        r3 = np.where(
            (np.abs(diff_x) > self["outlier"]) & (np.abs(diff_tt_x) < 5 * np.abs(diff_be_x)),
            1,
            0,
        )
        mask = r1 * np.abs(r2 * r3 - 1)
        self.mask_spots = mask

    def convert_selected(self):
        """Computes pixel geometry and intensity corrections for all unmasked pixels."""
        self.t0 = (
            np.array(
                [np.sin(np.radians(self["tiltx"])), np.sin(np.radians(self["tilty"])), 1.0],
                dtype=self.digit,
            )
            * self["distance"]
        )
        self.xyz_coordinates, self.normals = self.get_coordinates()
        if getattr(self, "mask", None) is None or not isinstance(self.mask, np.ndarray):
            self.mask = self.get_mask()
        r = np.where(self.mask != 0)

        coordinates = self.xyz_coordinates[r]
        beta = self.get_beta_radian(coordinates)
        rb = np.where(self.get_beta_range(beta) != 0)

        coordinates = coordinates[rb]
        tt = self.get_tt_radian(coordinates)
        rtt = np.where(self.get_tt_range(tt) != 0)
        q = self.radian_to_q(tt[rtt])
        rq = np.where(self.get_q_range(q) != 0)

        self.beta2d = beta[rb][rtt][rq]
        self.tt2d = tt[rtt][rq]
        self.q2d = q[rq]
        coordinates = coordinates[rtt][rq]

        self.sac = self.get_solid_angle_correction(coordinates)
        self.polarization = self.get_polar_corrction(self.tt2d, self.beta2d)
        self.absorption = self.get_absorption_correction(
            coordinates, self.normals[r][rb][rtt][rq]
        )
        self.air_absorption = self.get_air_absorption(coordinates)

        self.i2d = self.ibgs[r][rb][rtt][rq].astype(self.digit, copy=True)
        self.i2d *= self.sac / self.absorption / self.air_absorption
        self.i2d -= self["constant"]
        self.i2d /= self.polarization

    def convert_all(self):
        """Computes pixel geometry and intensity corrections across the entire detector grid."""
        self._logger.info("entire-region conversion starts...")
        self.t0 = (
            np.array(
                [np.sin(np.radians(self["tiltx"])), np.sin(np.radians(self["tilty"])), 1.0],
                dtype=self.digit,
            )
            * self["distance"]
        )
        self.xyz_coordinates, self.normals = self.get_coordinates()

        coords_flat = self.xyz_coordinates.reshape(-1, 3)
        normals_flat = self.normals.reshape(-1, 3)
        self.tt = self.get_tt_radian(coords_flat)
        self.beta = self.get_beta_radian(coords_flat)
        self.sac = self.get_solid_angle_correction(coords_flat)

        self.q = self.radian_to_q(self.tt)
        self.polarization = self.get_polar_corrction(self.tt, self.beta)
        self.absorption = self.get_absorption_correction(coords_flat, normals_flat)
        self.air_absorption = self.get_air_absorption(coords_flat)

        rq = self.get_q_range(self.q)
        rtt = self.get_tt_range(self.tt)
        rb = self.get_beta_range(self.beta)

        r = np.where(rq * rb * rtt * self.mask.flatten() != 0)
        self.i2d = self.ibgs.flatten()[r].astype(self.digit, copy=True)
        self.i2d *= self.sac[r] / self.absorption[r] / self.air_absorption[r]
        self.i2d -= self["constant"]
        self.i2d /= self.polarization[r]

        self.tt2d = self.tt[r]
        self.beta2d = self.beta[r]
        self.q2d = self.radian_to_q(self.tt2d)

    def get_coordinates(self):
        """Calculates 3D (x, y, z) coordinates and surface normal vectors for each pixel center."""
        y, x = np.mgrid[0 : self["pixy"], 0 : self["pixx"]]
        xy_coordinates = (
            np.array([x.flatten(), y.flatten()], dtype=self.digit).T.reshape(
                self["pixy"], self["pixx"], 2
            )
            + [0.5, 0.5]
        )
        xy_coordinates += [-self["x0"], -self["y0"]]
        xy_coordinates *= [self["pix_sizex"], self["pix_sizey"]]

        xa = xy_coordinates[:, :, 0]
        ya = xy_coordinates[:, :, 1]

        if self["curve_x"] == 0:
            x = xa
            z = np.zeros_like(x)
        else:
            x = np.sin(xa / self["curve_x"]) * self["curve_x"]
            z = self["curve_x"] - np.cos(xa / self["curve_x"]) * self["curve_x"]

        if self["curve_y"] == 0:
            y = ya
            z += np.zeros_like(y)
        else:
            y = np.sin(ya / self["curve_y"]) * self["curve_y"]
            z += self["curve_y"] - np.cos(ya / self["curve_y"]) * self["curve_y"]

        xyz_coordinates = np.concatenate(
            [x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]], axis=-1
        ).reshape(self["pixy"], self["pixx"], 3)
        grad_y, grad_x, _ = np.gradient(xyz_coordinates)
        normal = np.cross(grad_y, grad_x)

        return xyz_coordinates, normal

    def get_tt_radian(self, coordinates):
        """Calculates scattering angle 2theta (in radians) for the given pixel coordinates."""
        beam = coordinates - self.t0
        beam_length = np.linalg.norm(beam, axis=-1)
        beam_distance = np.linalg.norm(-self.t0)

        cos_tt = np.dot(beam, -self.t0) / (beam_length * beam_distance)
        tt_radian = np.arccos(np.clip(cos_tt, -1, 1))
        return tt_radian

    def get_beta_radian(self, coordinates):
        """Calculates azimuthal angle beta (in radians) for the given pixel coordinates."""
        beta0 = np.cross(self.t0, np.cross(np.array([1.0, 0.0, 0.0]), self.t0))
        beta_vectors = np.cross(self.t0, np.cross(coordinates, self.t0))
        radius = np.linalg.norm(beta_vectors, axis=-1)
        beta0_length = np.linalg.norm(beta0)

        cos_beta = np.dot(beta_vectors, beta0) / (radius * beta0_length)
        beta_radian = np.arccos(np.clip(cos_beta, -1, 1))

        ya = coordinates[:, 1]
        if self["curve_y"] == 0:
            y = ya
        else:
            y = np.sin(ya / self["curve_y"]) * self["curve_y"]
        polarity = np.where(y >= 0, 1.0, -1.0)
        return beta_radian * polarity

    def get_solid_angle_correction(self, coordinates):
        return self.solid_angle_correction(self.t0, coordinates, self)

    def get_polar_corrction(self, tt, beta):
        return self.polar_correction(tt, beta, self)

    def get_absorption_correction(self, coordinates, normals):
        return self.absorption_correction(self.t0, coordinates, normals, self)

    def get_air_absorption(self, coordinates):
        return self.air_absorption_correction(self.t0, coordinates, self)

    get_polar_correction = get_polar_corrction
    get_air_absorption_correction = get_air_absorption

    def get_q_range(self, q):
        rq = np.where((q < self["qmax"]) & (q > self["qmin"]), 1, 0)
        if (self["dqmin"] is not None) and (self["dqmax"] is not None):
            self.dq = np.average(np.abs(np.gradient(q)), axis=0)
            rdq = np.where((self.dq > self["dqmin"]) & (self.dq < self["dqmax"]), 1, 0)
        else:
            rdq = 1
        return rq * rdq

    def get_tt_range(self, tt):
        ttmax = self["ttmax"]
        ttmin = self["ttmin"]
        if isinstance(ttmax, (list, np.ndarray)):
            rtt = 0.0
            for n in range(len(ttmax)):
                rtt += np.where(
                    (np.radians(ttmin[n]) < tt) & (tt < np.radians(ttmax[n])), 1, 0
                )
        else:
            rtt = np.where((np.radians(ttmin) < tt) & (tt < np.radians(ttmax)), 1, 0)
        return rtt

    def get_beta_range(self, beta):
        def beta_round(be):
            hv = np.heaviside(be - np.pi, 0.0)
            be = be - np.pi * hv * 2
            return be, hv

        def beta_range(bmin, bmax):
            bmin, hmin = beta_round(bmin)
            bmax, hmax = beta_round(bmax)

            if hmin - hmax == 0:
                return np.where((bmin < beta) & (beta < bmax), 1, 0)
            elif hmin - hmax == -1:
                return np.abs(np.where((bmax < beta) & (beta < bmin), 1, 0) - 1)
            return np.ones_like(beta, dtype=int)

        betamax = self["betamax"]
        betamin = self["betamin"]
        if isinstance(betamax, (list, np.ndarray)):
            bmax_list = np.atleast_1d(betamax)
            bmin_list = np.atleast_1d(betamin)
            rb = 0.0
            for n in range(len(bmax_list)):
                rb += beta_range(np.radians(bmin_list[n]), np.radians(bmax_list[n]))
        else:
            rb = beta_range(np.radians(betamin), np.radians(betamax))
        return np.clip(rb, 0, 1)

    def set_azimuthal_mode(
        self,
        mode: str = "full",
        width: float = 20.0,
        center_angles: list[float] | None = None,
        betamin: float | list[float] | None = None,
        betamax: float | list[float] | None = None,
    ):
        """Configure azimuthal integration sectors for detector geometry and polarization.

        Parameters
        ----------
        mode : {"full", "diagonals", "corners", "vertical", "horizontal", "custom", "manual"}
            Azimuthal sector configuration:
            - "full": 360-degree integration (beta in [-180, 180]).
            - "diagonals" or "corners": 4 diagonal sectors centered at +/-45 and +/-135 deg,
              avoiding flat-panel detector edge cutoffs at high Q.
            - "vertical" or "up-down": 2 vertical sectors centered at +/-90 deg, maximizing synchrotron
              horizontal polarization purity (P = 1.0).
            - "horizontal" or "left-right": 2 horizontal sectors centered at 0 and 180 deg.
            - "custom": custom sectors specified by `center_angles` +/- width.
            - "manual": arbitrary manual sectors specified directly by `betamin` and `betamax`.
        width : float, optional
            Half-width of the angular sectors in degrees (default: 20.0).
        center_angles : list of float, optional
            List of sector center angles (in degrees) when mode is "custom".
        betamin : float or list of float, optional
            Explicit lower beta angle bounds (in degrees) when mode is "manual".
        betamax : float or list of float, optional
            Explicit upper beta angle bounds (in degrees) when mode is "manual".

        Returns
        -------
        Pixia
            Self instance for method chaining.
        """
        mode = mode.lower()
        self.azimuthal_mode = mode
        self.azimuthal_width = width
        if mode in ("full", "360", "all"):
            self["betamin"] = -180.0
            self["betamax"] = 180.0
        elif mode in ("diagonals", "corners", "diagonal"):
            centers = [45.0, 135.0, -135.0, -45.0]
            self["betamin"] = [c - width for c in centers]
            self["betamax"] = [c + width for c in centers]
        elif mode in ("vertical", "up-down", "updown", "top-bottom"):
            centers = [90.0, -90.0]
            self["betamin"] = [c - width for c in centers]
            self["betamax"] = [c + width for c in centers]
        elif mode in ("horizontal", "left-right", "leftright"):
            centers = [0.0, 180.0]
            self["betamin"] = [c - width for c in centers]
            self["betamax"] = [c + width for c in centers]
        elif mode == "custom":
            if center_angles is None:
                raise ValueError("`center_angles` must be provided when mode is 'custom'")
            self["betamin"] = [c - width for c in center_angles]
            self["betamax"] = [c + width for c in center_angles]
        elif mode == "manual":
            if betamin is None or betamax is None:
                raise ValueError("`betamin` and `betamax` must be provided when mode is 'manual'")
            self["betamin"] = betamin
            self["betamax"] = betamax
        else:
            raise ValueError(f"Unknown azimuthal mode: {mode}")
        return self

    set_integration_sector = set_azimuthal_mode

    def get_azimuthal_mode_summary(self) -> str:
        """Returns a human-readable summary of the current azimuthal integration sector."""
        bmin = getattr(self, "par", {}).get("betamin", {}).get("value", None)
        bmax = getattr(self, "par", {}).get("betamax", {}).get("value", None)
        if bmin is None or bmax is None:
            return "Full (360° unconstrained)"
        bmin_arr = np.atleast_1d(bmin)
        bmax_arr = np.atleast_1d(bmax)
        if len(bmin_arr) == 1 and np.isclose(bmin_arr[0], -180.0) and np.isclose(bmax_arr[0], 180.0):
            return "Full (360° unconstrained: -180° to 180°)"
        if len(bmin_arr) == 4 and len(bmax_arr) == 4:
            centers = (bmin_arr + bmax_arr) / 2.0
            if np.allclose(centers, [45.0, 135.0, -135.0, -45.0], atol=1.0):
                w = float((bmax_arr[0] - bmin_arr[0]) / 2.0)
                return f"Diagonals: ±45°, ±135° (Δβ = ±{w:.1f}°)"
        if len(bmin_arr) == 2 and len(bmax_arr) == 2:
            centers = (bmin_arr + bmax_arr) / 2.0
            if np.allclose(centers, [90.0, -90.0], atol=1.0):
                w = float((bmax_arr[0] - bmin_arr[0]) / 2.0)
                return f"Vertical: ±90° (Δβ = ±{w:.1f}°, Synchrotron Horiz. Pol.)"
            elif np.allclose(centers, [0.0, 180.0], atol=1.0):
                w = float((bmax_arr[0] - bmin_arr[0]) / 2.0)
                return f"Horizontal: 0°, 180° (Δβ = ±{w:.1f}°)"
        bmin_str = ", ".join(f"{x:.1f}" for x in bmin_arr) if len(bmin_arr) > 1 else f"{bmin_arr[0]:.1f}"
        bmax_str = ", ".join(f"{x:.1f}" for x in bmax_arr) if len(bmax_arr) > 1 else f"{bmax_arr[0]:.1f}"
        return f"Manual β: min=[{bmin_str}], max=[{bmax_str}]"

    def get_detector_edge_q(self):
        """Calculate the physical edge cutoff Q and diagonal corner Q limits of the rectangular detector.

        Returns
        -------
        dict
            Dictionary containing:
            - 'q_edge': Maximum Q reached on the inscribed circle (detector edge).
            - 'q_corner': Maximum Q reachable in the extreme diagonal corners.
            - 'twotheta_edge': 2theta at detector edge in degrees.
            - 'twotheta_corner': 2theta at diagonal corner in degrees.
        """
        x0, y0 = self["x0"], self["y0"]
        psx, psy = self["pix_sizex"], self["pix_sizey"]
        dist = self["distance"]
        lam = self["rambda"]

        d_left = x0 * psx
        d_right = (self["pixx"] - x0) * psx
        d_bottom = y0 * psy
        d_top = (self["pixy"] - y0) * psy

        d_edge_min = min(d_left, d_right, d_bottom, d_top)
        d_corner_max = np.sqrt(max(d_left, d_right) ** 2 + max(d_bottom, d_top) ** 2)

        tt_edge = np.arctan(d_edge_min / dist)
        tt_corner = np.arctan(d_corner_max / dist)

        q_edge = 4.0 * np.pi / lam * np.sin(tt_edge / 2.0)
        q_corner = 4.0 * np.pi / lam * np.sin(tt_corner / 2.0)

        return {
            "q_edge": float(q_edge),
            "q_corner": float(q_corner),
            "twotheta_edge": float(np.degrees(tt_edge)),
            "twotheta_corner": float(np.degrees(tt_corner)),
        }

    def auto_azimuthal_mode(self, target_qmax: float | None = None, width: float = 20.0):
        """Automatically select optimal azimuthal integration sectors based on detector geometry and target Qmax.

        If target_qmax exceeds the inscribed circle edge limit (Q_edge), this method
        automatically activates diagonal corner sectors ('diagonals') to prevent abrupt
        geometric truncation discontinuities. If target_qmax <= Q_edge, full integration
        is retained for maximum photon counting statistics.

        Parameters
        ----------
        target_qmax : float, optional
            Target maximum Q (in Angstrom^-1). If None, inferred from self['ttmax'] or self['qmax'].
        width : float, optional
            Angular half-width for sector selection in degrees (default: 20.0).

        Returns
        -------
        str
            Selected azimuthal mode ("diagonals" or "full").
        """
        limits = self.get_detector_edge_q()
        q_edge = limits["q_edge"]

        if target_qmax is None:
            if self["qmax"] is not None:
                target_qmax = self["qmax"]
            else:
                tt_max_rad = (
                    np.radians(self["ttmax"])
                    if not isinstance(self["ttmax"], (list, np.ndarray))
                    else np.radians(np.max(self["ttmax"]))
                )
                target_qmax = 4.0 * np.pi / self["rambda"] * np.sin(tt_max_rad / 2.0)

        if target_qmax > q_edge * 0.98:
            self._logger.info(
                f"Target Qmax ({target_qmax:.2f} A^-1) exceeds detector edge limit ({q_edge:.2f} A^-1). "
                "Automatically selecting diagonal corner sectors to avoid edge truncation artifacts."
            )
            self.set_azimuthal_mode("diagonals", width=width)
            return "diagonals"
        else:
            self.set_azimuthal_mode("full")
            return "full"

    def twotheta_intensity(self, rmse=False):
        """Integrates corrected 2D pixel intensities onto a 1D 2theta grid with optional outlier rejection."""
        self._logger.info("twotheta_intensity start...")
        tt_deg = np.degrees(self.tt2d)
        self.tt1d, self.i1d = self.get_1d_profile(tt_deg, self.i2d)

        if self["outlier"] != 0.0:
            tt, i = self.outlier_rejection(tt_deg, self.i2d, self.tt1d, self.i1d)
            self.tt1d, self.i1d = self.get_1d_profile(tt, i)
        else:
            tt, i = tt_deg, self.i2d

        if rmse:
            self.rmse = self.get_rmse(tt, i, self.tt1d, self.i1d)

    def get_1d_profile(self, tt, i):
        """Bins 2D pixel data (tt, i) onto the uniform 1D 2theta grid."""
        x_new = np.arange(self["ttmin"], self["ttmax"] + self["ttstep"], self["ttstep"])
        i_out = profile_calculation(
            np.ascontiguousarray(tt.flatten(), dtype=np.float64),
            np.ascontiguousarray(i.flatten(), dtype=np.float64),
            np.ascontiguousarray(x_new, dtype=np.float64),
            verbose=False,
        )
        return x_new, i_out

    def outlier_rejection(self, tt2d, i2d, tt1d, i1d):
        """Rejects pixels whose intensities deviate from the azimuthally averaged 1D profile beyond `self['outlier']`."""
        i_ave = intp.interp1d(tt1d, i1d, kind="linear", bounds_error=False, fill_value=0)(tt2d)
        r = i_ave.nonzero()
        ratio = i2d[r] / i_ave[r]
        r2 = np.where((ratio < 1.0 + self["outlier"]) & (ratio > 1.0 - self["outlier"]))

        tt_out = tt2d[r][r2]
        i_out = i2d[r][r2]
        return tt_out, i_out

    def get_rmse(self, tt2d, i2d, tt1d, i1d):
        min_rad = np.radians(self["ttmin"])
        max_rad = np.radians(self["ttmax"])
        step_rad = np.radians(self["ttstep"])
        x_new = np.arange(min_rad, max_rad + step_rad * 2.0, step_rad)
        rmse_ave = np.interp(x_new, tt1d, i1d, left=0, right=0)
        _, i_out = rmse_calculation(tt2d.flatten(), i2d.flatten(), x_new, rmse_ave)
        return i_out

    def radian_to_q(self, tt):
        return 4.0 * np.pi / self["rambda"] * np.sin(tt / 2.0)

    def file_mtime_ordering(self, files):
        new_files = []
        if not isinstance(files, list):
            files = [files]
        for file in files:
            if "*" in file:
                new_files += glob.glob(file)
            else:
                new_files += [file]
        new_files = sorted(new_files, key=os.path.getmtime)
        return new_files

    def get_background_curve(self):
        """Compute the 1D dummy-frame baseline profile.

        Propagates solid angle, sensor absorption, air attenuation, and polarization
        corrections to create an empirical baseline array for downstream PDFget reduction.

        Returns
        -------
        numpy.ndarray
            1D baseline array ``d1d`` evaluated on the integration grid.
        """
        self.dummy2d()
        self.dummy1d()
        return self.d1d

    def dummy2d(self):
        scale = self.sac / self.absorption / self.air_absorption / self.polarization
        self.d2d = scale * 10.0

    def dummy1d(self):
        _, self.d1d = self.get_1d_profile(np.degrees(self.tt2d), self.d2d)

    def calibrate(
        self,
        standard="ceo2",
        n_trials=(80, 80, 20, 20),
        scales=(1.0, 1.0, 0.2, 0.2),
        fit_keys=None,
        mode="pn",
        engine="lmfit",
    ):
        """Perform multi-stage geometry calibration using standard powder rings.

        Parameters
        ----------
        standard : str, optional
            Calibrant material name (e.g. "ceo2", "ni"). Default is "ceo2".
        n_trials : tuple of int, optional
            Number of Optuna trials per refinement stage (when using Optuna engines). Default is (80, 80, 20, 20).
        scales : tuple of float, optional
            Exploration scales for parameter search bounds. Default is (1.0, 1.0, 0.2, 0.2).
        fit_keys : list of str, optional
            Explicit list of parameter keys to refine (e.g. ["x0", "y0", "distance"]).
            If None, respects `par[k]["optimize"]` configuration.
        mode : str, optional
            Calibration mode: "pn" (opposing ±X, ±Y for centered beam) or
            corner modes like "xp-yp" (+X vs +Y for corner beam), "xn-yp", "xp-yn", "xn-yn".
            Default is "pn".
        engine : str, optional
            Optimization engine: "lmfit" (Non-linear Least Squares / Levenberg-Marquardt),
            "cmaes" (Optuna CMA-ES), "tpe" (Optuna TPE), or "nsgaii" (Optuna NSGA-II).
            Default is "lmfit".

        Returns
        -------
        Pixia
            Self instance with optimized detector distance, center, and tilt angles.
        """
        from .calibrator import Calibrator

        if not isinstance(getattr(self, "ibgs", None), np.ndarray):
            if not isinstance(getattr(self, "sample", None), np.ndarray):
                self.read()
            self.bg_subtraction()
        if getattr(self, "mask", None) is None or not isinstance(self.mask, np.ndarray):
            self.mask = self.get_mask()

        # Ensure natural synchrotron bounds around current values for active keys
        for k in (fit_keys or ["x0", "y0", "tiltx", "tilty", "distance", "rambda"]):
            if k in self.par and self[k] is not None:
                val = float(np.ravel(self[k])[0])
                p_min = self.par[k].get("min", [None])[0]
                p_max = self.par[k].get("max", [None])[0]
                if p_min is None or p_max is None or p_min >= p_max or val < p_min or val > p_max or (p_max - p_min) > max(15.0, abs(val) * 0.3):
                    if k in ("x0", "y0"):
                        self.par[k]["min"] = [float(np.round(val - 5.0, 2))]
                        self.par[k]["max"] = [float(np.round(val + 5.0, 2))]
                    elif k in ("tiltx", "tilty"):
                        self.par[k]["min"] = [float(np.round(val - 1.0, 3))]
                        self.par[k]["max"] = [float(np.round(val + 1.0, 3))]
                    elif k == "distance":
                        self.par[k]["min"] = [float(np.round(val * 0.95, 3))]
                        self.par[k]["max"] = [float(np.round(val * 1.05, 3))]
                    elif k == "rambda":
                        self.par[k]["min"] = [float(np.round(val * 0.97, 5))]
                        self.par[k]["max"] = [float(np.round(val * 1.03, 5))]

        self._state = "CALIBRATING"
        cal = Calibrator(pixia=self, standard=standard)
        use_sparse_mask = str(engine).lower() not in ("lmfit", "lls", "leastsq")
        if use_sparse_mask:
            cal.set_mask()
        cal.optimize(
            n_trials=list(n_trials),
            scales=list(scales),
            targets=["center", "peak_angle", "center", "peak_angle"],
            key_diff={
                "rambda": 0.01,
                "x0": 2.0,
                "y0": 2.0,
                "tiltx": 0.5,
                "tilty": 0.5,
                "distance": 10.0,
            },
            fit_keys=fit_keys,
            mode=mode,
            engine=engine,
        )
        if use_sparse_mask:
            cal.reset_mask()
        self.convert_selected()
        self.twotheta_intensity()
        self.get_background_curve()
        self._state = "CALIBRATED"
        return self

    def set_natural_bounds(
        self,
        rambda_pct: float = 0.03,
        distance_pct: float = 0.05,
        tilt_deg: float = 1.0,
        center_px: float = 5.0,
    ):
        """Set realistic synchrotron parameter exploration bounds (min, max) around current values.

        Parameters
        ----------
        rambda_pct : float, optional
            Relative wavelength tolerance (default: 0.03, corresponding to ±3%).
        distance_pct : float, optional
            Relative sample-to-detector distance tolerance (default: 0.05, corresponding to ±5%).
        tilt_deg : float, optional
            Maximum angular detector tilt tolerance in degrees (default: 1.0, corresponding to ±1.0°).
        center_px : float, optional
            Direct beam center tolerance in pixels (default: 5.0, corresponding to ±5 px).
        """
        if self["rambda"] is not None:
            lam = float(np.ravel(self["rambda"])[0])
            self.par["rambda"]["min"] = [float(np.round(lam * (1.0 - rambda_pct), 5))]
            self.par["rambda"]["max"] = [float(np.round(lam * (1.0 + rambda_pct), 5))]

        if self["distance"] is not None:
            dist = float(np.ravel(self["distance"])[0])
            self.par["distance"]["min"] = [float(np.round(dist * (1.0 - distance_pct), 3))]
            self.par["distance"]["max"] = [float(np.round(dist * (1.0 + distance_pct), 3))]

        if self["tiltx"] is not None:
            tx = float(np.ravel(self["tiltx"])[0])
            self.par["tiltx"]["min"] = [float(np.round(tx - tilt_deg, 3))]
            self.par["tiltx"]["max"] = [float(np.round(tx + tilt_deg, 3))]
        if self["tilty"] is not None:
            ty = float(np.ravel(self["tilty"])[0])
            self.par["tilty"]["min"] = [float(np.round(ty - tilt_deg, 3))]
            self.par["tilty"]["max"] = [float(np.round(ty + tilt_deg, 3))]

        if self["x0"] is not None:
            x0 = float(np.ravel(self["x0"])[0])
            self.par["x0"]["min"] = [float(np.round(x0 - center_px, 2))]
            self.par["x0"]["max"] = [float(np.round(x0 + center_px, 2))]
        if self["y0"] is not None:
            y0 = float(np.ravel(self["y0"])[0])
            self.par["y0"]["min"] = [float(np.round(y0 - center_px, 2))]
            self.par["y0"]["max"] = [float(np.round(y0 + center_px, 2))]

    def get_directional_profiles(self, width: float = 10.0, standard: str | None = None, mode: str = "pn") -> dict:
        """Extract directional 1D profiles (+X, -X, +Y, -Y, diagonals) and theoretical Bragg peaks."""
        from .calibrator import Calibrator

        cal = Calibrator(pixia=self, standard=standard or "ni")
        return cal.get_directional_profiles(width=width, standard=standard, mode=mode)

    def get_standard_bragg_reflections(self, standard: str = "ni") -> dict:
        """Compute theoretical Bragg peak positions 2theta and detector radii for standard calibrant."""
        from .calibrator import Calibrator

        cal = Calibrator(pixia=self, standard=standard)
        prof = cal.get_directional_profiles(width=5.0, standard=standard)
        return {
            "standard": standard,
            "tt_deg": prof["std_bragg_tt"],
            "radius_mm": prof["std_bragg_radius"],
            "intensity": prof["std_bragg_intensity"],
            "d_spacing": prof["std_bragg_d"],
        }

    def auto_beamstop_mask(self, search_box: int = 100, threshold_factor: float = 0.15) -> int:
        """Automatically detect direct beamstop shadow circle around nominal beam center."""
        from .calibrator import Calibrator

        cal = Calibrator(pixia=self)
        return cal.auto_beamstop_mask(search_box=search_box, threshold_factor=threshold_factor)

    def align_center_tilt(
        self, width: float = 15.0, max_iter: int = 30, xatol: float = 0.02, fatol: float = 1e-4
    ) -> dict:
        """Align beam center (x0, y0) and detector tilts (tiltx, tilty) via opposing sector symmetry."""
        from .calibrator import Calibrator

        cal = Calibrator(pixia=self)
        return cal.align_center_tilt(width=width, max_iter=max_iter, xatol=xatol, fatol=fatol)

    def tune_polarization(self, width: float = 15.0) -> float:
        """Tune synchrotron horizontal polarization factor p using pristine vertical reference sectors."""
        from .calibrator import Calibrator

        cal = Calibrator(pixia=self)
        return cal.tune_polarization(width=width)

    def calibrate_distance_with_pdf(
        self,
        cif_file: str | None = None,
        standard: str = "ni",
        target_lattice: float = 3.5238,
        qmax: float = 20.0,
        work_dir: str | None = None,
    ) -> dict:
        """Calibrate sample-to-detector distance and wavelength via standard reference PDF refinement."""
        from .calibrator import Calibrator

        cal = Calibrator(pixia=self, standard=standard)
        return cal.calibrate_distance_wavelength_with_pdf(
            cif_file=cif_file,
            standard=standard,
            target_lattice=target_lattice,
            qmax=qmax,
            work_dir=work_dir,
        )

    def auto_calibrate(
        self,
        cif_file: str | None = None,
        standard: str = "ni",
        target_lattice: float = 3.5238,
        qmax: float = 20.0,
        work_dir: str | None = None,
        fix_wavelength: bool = False,
        fix_distance: bool = False,
    ) -> dict:
        """Execute the complete multi-stage automated 2D scattering calibration pipeline."""
        try:
            from orochi.pipeline import CalibrationPipeline
            pipe = CalibrationPipeline(
                standard=standard,
                target_lattice=target_lattice,
                qmax=qmax,
                work_dir=work_dir,
            )
            return pipe.run(
                self,
                fix_wavelength=fix_wavelength,
                fix_distance=fix_distance,
                run_pdf_instrumental=True,
            )
        except ImportError:
            from .calibrator import Calibrator
            cal = Calibrator(pixia=self, standard=standard)
            return cal.auto_pipeline(
                cif_file=cif_file,
                standard=standard,
                target_lattice=target_lattice,
                qmax=qmax,
                work_dir=work_dir,
            )


    def explain(self):
        """Generate a structured Markdown diagnostic summary of the 2D detector integration.

        Returns
        -------
        str
            Multi-line Markdown report covering detector geometry, beam center,
            radiation parameters, active mask fraction, and 1D profile status.
        """
        lines = [
            "# 2D Image Reduction Report (`orochi.pixia`)",
            f"- **Detector Grid**: `{self['pixx']} x {self['pixy']}` pixels ($\\Delta x = {self['pix_sizex']}$ mm, $\\Delta y = {self['pix_sizey']}$ mm)",
            f"- **Beam Center & Geometry**: $(x_0, y_0) = ({self['x0']:.2f}, {self['y0']:.2f})$ px, distance $L_0 = {self['distance']:.3f}$ mm, tilts $(\\theta_x, \\theta_y) = ({self['tiltx']:.3f}^\\circ, {self['tilty']:.3f}^\\circ)$",
            f"- **Radiation & Sensor**: $\\lambda = {self['rambda']:.5f}$ A ($E = {self['energy']:.3f}$ keV), polarization $P = {self['polar']:.3f}$, sensor thickness $t = {self['thickness']}$ cm, $\\mu = {self['mur']}$ cm$^{{-1}}$",
            f"- **Integration Range**: $2\\theta \\in [{self['ttmin']:.2f}^\\circ, {self['ttmax']:.2f}^\\circ]$ ($\\Delta(2\\theta) = {self['ttstep']}^\\circ$), outlier threshold $\\epsilon = {self['outlier']}$",
        ]
        if self.mask is not None:
            active_frac = float(np.mean(self.mask != 0)) * 100.0
            lines.append(f"- **Active Unmasked Pixels**: `{active_frac:.1f}%`")
        if self.tt1d is not None and self.i1d is not None:
            lines.append(
                f"- **1D Profile Generated**: `{len(self.tt1d)}` points, dummy baseline propagated: `{self.d1d is not None}`"
            )
        return "\n".join(lines)


# Backward-compatible aliases for legacy Orochi 3 scripts
par = Pixia


def calculator(pixia_par=None):
    """Legacy factory returning a configured ``Pixia`` instance.

    Parameters
    ----------
    pixia_par : Pixia, optional
        Existing Pixia instance or parameter container. If None, a new
        Pixia instance is created.

    Returns
    -------
    Pixia
        Configured Pixia instance.
    """
    return pixia_par if isinstance(pixia_par, Pixia) else Pixia()


import warnings
import numpy as np
try:
    import optuna
except ImportError:
    optuna = None
from scipy import ndimage as ndim
from scipy import interpolate as intp
from scipy.optimize import minimize

from orochi.pixia.calculator.profile import profile_calculation
from orochi.pixia.database import standard_diffraction

warnings.simplefilter("ignore")


class Calibrator:
    """Multi-objective Optuna geometry calibrator for 2D diffraction detectors in `orochi.pixia`."""

    def __init__(self, pixia=None, standard="ceo2"):
        self.pixia = pixia
        self.standard = standard
        self.peak_width = 0.1  # in Q [A^-1]
        self.q_min = 0.0  # in Q [A^-1]
        self.q_max = 25.0  # in Q [A^-1]

    def optimize(
        self,
        n_trials=[20, 20, 20, 20],
        scales=[1.0, 1.0, 0.2, 0.2],
        targets=["peak_width", "peak_angle", "peak_width", "peak_angle"],
        key_diff={
            "rambda": 0.01,
            "x0": 2.0,
            "y0": 2.0,
            "tiltx": 0.5,
            "tilty": 0.5,
            "distance": 10.0,
        },
        fit_keys=None,
        mode="pn",
        engine="lmfit",
    ):
        """Runs geometry calibration using either Non-linear Least Squares (lmfit) or Bayesian optimization (Optuna)."""
        if fit_keys is not None:
            active_keys = set(fit_keys)
        else:
            active_keys = {
                k
                for k in ["x0", "y0", "tiltx", "tilty", "distance", "rambda"]
                if getattr(self.pixia, "par", {}).get(k, {}).get("optimize", False)
            }
            if not active_keys:
                # Default fallback if called programmatically without GUI flags
                active_keys = {"x0", "y0", "tiltx", "tilty", "distance"}

        if self.pixia is not None:
            self.pixia._state = "CALIBRATING"

        engine_str = str(engine).lower()
        if engine_str in ("lmfit", "lls", "leastsq"):
            cand = [k for k in ["x0", "y0", "tiltx", "tilty", "distance", "rambda"] if k in active_keys]
            if cand:
                self._optimize_lmfit(cand=cand, active_keys=active_keys, mode=mode)
            self.pixia.last_calibrated_keys = list(active_keys)
            self.pixia.last_calibrated_mode = mode
            self.pixia.last_calibrated_engine = "lmfit"
            if self.pixia is not None:
                self.pixia._state = "CALIBRATED"
            return

        def _create_sampler():
            if "cma" in engine_str:
                return optuna.samplers.CmaEsSampler()
            elif "tpe" in engine_str:
                return optuna.samplers.TPESampler()
            else:
                return optuna.samplers.NSGAIISampler()

        def _diff_val(val):
            return float(val[0]) if isinstance(val, (list, tuple, np.ndarray)) else float(val)

        def _suggest_val(trial, key, init_val, d):
            par_entry = getattr(self.pixia, "par", {}).get(key, {})
            p_min_raw = par_entry.get("min", None)
            p_max_raw = par_entry.get("max", None)
            p_min = float(p_min_raw[0] if isinstance(p_min_raw, (list, tuple, np.ndarray)) else (p_min_raw if p_min_raw is not None else -np.inf))
            p_max = float(p_max_raw[0] if isinstance(p_max_raw, (list, tuple, np.ndarray)) else (p_max_raw if p_max_raw is not None else np.inf))
            val = float(init_val)
            if p_min >= p_max or val < p_min or val > p_max:
                if key in ("x0", "y0"):
                    p_min, p_max = val - 5.0, val + 5.0
                elif key in ("tiltx", "tilty"):
                    p_min, p_max = val - 1.0, val + 1.0
                elif key == "distance":
                    p_min, p_max = val * 0.95, val * 1.05
                elif key == "rambda":
                    p_min, p_max = val * 0.97, val * 1.03
                else:
                    p_min, p_max = val - abs(val) * 0.05, val + abs(val) * 0.05
                if key in getattr(self.pixia, "par", {}):
                    self.pixia.par[key]["min"] = [float(p_min)]
                    self.pixia.par[key]["max"] = [float(p_max)]
            low = max(val - d, p_min)
            high = min(val + d, p_max)
            if low >= high:
                low = val - abs(d)
                high = val + abs(d)
            return trial.suggest_float(key, low, high)

        mode_str = str(mode).lower()
        is_corner_mode = mode_str not in ("pn", "xn-xp,yn-yp", "centered")

        if is_corner_mode:
            # Corner beam geometry mode (e.g. xp-yp, xn-yp, xp-yn, xn-yn)
            if "xn-yp" in mode_str:
                a1, a2, ad = 180.0, 90.0, 135.0
            elif "xp-yn" in mode_str:
                a1, a2, ad = 0.0, -90.0, -45.0
            elif "xn-yn" in mode_str:
                a1, a2, ad = 180.0, -90.0, -135.0
            else:  # "xp-yp"
                a1, a2, ad = 0.0, 90.0, 45.0

            cand = [k for k in ["x0", "y0", "tiltx", "tilty", "distance", "rambda"] if k in active_keys]
            if not cand:
                return

            std_ref = self.pixia.get_standard_bragg_reflections(self.standard)
            peaks_tt = std_ref["tt_deg"]
            if len(peaks_tt) > 8:
                peaks_tt = peaks_tt[:8]

            step = 2
            if getattr(self.pixia, "mask", None) is None:
                self.pixia.mask = self.pixia.get_mask()
            mask_sub = (self.pixia.mask != 0)[::step, ::step]
            ibgs_sub = self.pixia.ibgs[::step, ::step].astype(np.float32)
            y_sub, x_sub = np.mgrid[0 : self.pixia["pixy"] : step, 0 : self.pixia["pixx"] : step]
            x_sub = x_sub.astype(np.float32)
            y_sub = y_sub.astype(np.float32)
            psx = float(self.pixia["pix_sizex"])
            psy = float(self.pixia["pix_sizey"])

            ttmin = float(self.pixia["ttmin"]) if self.pixia["ttmin"] is not None else 1.0
            ttmax = float(self.pixia["ttmax"]) if self.pixia["ttmax"] is not None else 30.0
            ttstep = max(float(self.pixia["ttstep"]) if self.pixia["ttstep"] is not None else 0.01, 0.005)
            x_grid = np.arange(ttmin, ttmax + ttstep, ttstep, dtype=np.float64)

            def _get_peaks(x_arr, y_arr):
                pks = []
                for pt in peaks_tt:
                    m = (x_arr >= pt - 0.4) & (x_arr <= pt + 0.4)
                    if np.count_nonzero(m) < 5:
                        continue
                    sub_x, sub_y = x_arr[m], y_arr[m]
                    idx = np.argmax(sub_y)
                    k0 = max(0, idx - 4)
                    k1 = min(len(sub_x), idx + 5)
                    if k1 - k0 >= 5:
                        bg_line = np.linspace(sub_y[k0], sub_y[k1 - 1], k1 - k0)
                        fit_p = np.polyfit(sub_x[k0:k1], sub_y[k0:k1] - bg_line, 2)
                        if fit_p[0] < 0:
                            pks.append(-fit_p[1] / (2.0 * fit_p[0]))
                            continue
                    pks.append(sub_x[idx])
                return np.asarray(pks)

            def objective_corner_fn(opt_init, opt_diff):
                def obj(trial):
                    for k in cand:
                        d = _diff_val(opt_diff[k])
                        self.pixia[k] = _suggest_val(trial, k, opt_init[k], d)

                    dist = float(self.pixia["distance"])
                    t0 = np.array(
                        [np.sin(np.radians(self.pixia["tiltx"])), np.sin(np.radians(self.pixia["tilty"])), 1.0],
                        dtype=np.float32,
                    ) * dist
                    dx = (x_sub - float(self.pixia["x0"])) * psx
                    dy = (y_sub - float(self.pixia["y0"])) * psy
                    beta = np.degrees(np.arctan2(dy, dx))

                    bx = dx - t0[0]
                    by = dy - t0[1]
                    bz = -t0[2]
                    beam_len = np.sqrt(bx * bx + by * by + bz * bz)
                    t0_len = np.linalg.norm(t0)
                    dot = -(bx * t0[0] + by * t0[1] + bz * t0[2])
                    cos_tt = np.clip(dot / (beam_len * t0_len), -1.0, 1.0)
                    tt_deg = np.degrees(np.arccos(cos_tt))

                    width = 10.0
                    def _sec_idx(target_ang):
                        diff = np.abs(beta - target_ang)
                        diff = np.minimum(diff, 360.0 - diff)
                        return mask_sub & (diff < width)

                    idx1 = _sec_idx(a1)
                    idx2 = _sec_idx(a2)
                    if not np.any(idx1) or not np.any(idx2):
                        return 100.0

                    i1 = profile_calculation(
                        np.ascontiguousarray(tt_deg[idx1], dtype=np.float64),
                        np.ascontiguousarray(ibgs_sub[idx1], dtype=np.float64),
                        x_grid,
                        False,
                    )
                    i2 = profile_calculation(
                        np.ascontiguousarray(tt_deg[idx2], dtype=np.float64),
                        np.ascontiguousarray(ibgs_sub[idx2], dtype=np.float64),
                        x_grid,
                        False,
                    )

                    p1 = _get_peaks(x_grid, i1)
                    p2 = _get_peaks(x_grid, i2)
                    n = min(len(p1), len(p2))
                    if n == 0:
                        return 50.0

                    e_cross = float(np.mean((p1[:n] - p2[:n]) ** 2))
                    e_bragg = float(0.5 * (np.mean((p1[:n] - peaks_tt[:n]) ** 2) + np.mean((p2[:n] - peaks_tt[:n]) ** 2)))
                    max1 = np.nanmax(i1) if np.nanmax(i1) > 0 else 1.0
                    max2 = np.nanmax(i2) if np.nanmax(i2) > 0 else 1.0
                    e_shape = float(np.mean((i1 / max1 - i2 / max2) ** 2))

                    return 10.0 * e_cross + 5.0 * e_bragg + 0.1 * e_shape

                return obj

            corner_trials = [sum(n_trials[:2]), sum(n_trials[2:])] if len(n_trials) >= 4 else [50, 30]
            corner_scales = [1.0, 0.2]
            for st_idx, (nt, sc) in enumerate(zip(corner_trials, corner_scales)):
                if nt <= 0:
                    continue
                opt_diff = {k: key_diff.get(k, 0.1) * sc for k in cand}
                opt_init = {k: float(self.pixia[k]) for k in cand}

                print(f"\nstep {st_idx + 1} : corner geometry optimization (mode={mode}, scale={sc}) {cand}")
                study = optuna.create_study(
                    directions=["minimize"], sampler=_create_sampler()
                )
                study.enqueue_trial(opt_init)
                study.optimize(objective_corner_fn(opt_init, opt_diff), n_trials=nt)
                for key, val in study.best_params.items():
                    self.pixia[key] = val

        else:
            # Centered opposing symmetry mode ("pn" / "xn-xp,yn-yp")
            def objective_peak_angle(opt_init, opt_diff):
                def obj(trial):
                    for key in opt_init.keys():
                        d = _diff_val(opt_diff[key])
                        self.pixia[key] = _suggest_val(trial, key, opt_init[key], d)
                    tt1d, i1d = self.sparse_data()
                    _, peak_angle, _ = self.peak_analysis(tt1d, i1d)
                    return float(np.average(np.square(peak_angle)))

                return obj

            def objective_peak_width(opt_init, opt_diff):
                def obj(trial):
                    for key in opt_init.keys():
                        d = _diff_val(opt_diff[key])
                        self.pixia[key] = _suggest_val(trial, key, opt_init[key], d)
                    tt1d, i1d = self.sparse_data()
                    _, _, peak_width = self.peak_analysis(tt1d, i1d)
                    return float(np.average(peak_width))

                return obj

            def objective_center(opt_init, opt_diff):
                def obj(trial):
                    for key in opt_init.keys():
                        d = _diff_val(opt_diff[key])
                        self.pixia[key] = _suggest_val(trial, key, opt_init[key], d)
                    d_list = self.center_analysis()
                    return float(np.average(d_list))

                return obj

            for index, nt in enumerate(n_trials):
                target = targets[index]
                opt_diff = {}
                opt_init = {}

                if target in ("peak_width", "center"):
                    cand = [k for k in ["x0", "y0", "tiltx", "tilty"] if k in active_keys]
                    if not cand:
                        continue
                    for k in cand:
                        opt_diff[k] = key_diff[k] * scales[index]
                        opt_init[k] = float(self.pixia[k])

                    print(f"\nstep {index + 1} : optimization for {target} {cand}")
                    study = optuna.create_study(
                        directions=["minimize"], sampler=_create_sampler()
                    )
                    study.enqueue_trial(opt_init)
                    obj_fn = objective_peak_width(opt_init, opt_diff) if target == "peak_width" else objective_center(opt_init, opt_diff)
                    study.optimize(obj_fn, n_trials=nt)
                    for key, val in study.best_params.items():
                        self.pixia[key] = val

                elif target == "peak_angle":
                    cand = [k for k in ["rambda", "distance"] if k in active_keys]
                    if not cand:
                        continue
                    for k in cand:
                        opt_diff[k] = key_diff[k] * scales[index]
                        opt_init[k] = float(self.pixia[k])

                    print(f"\nstep {index + 1} : optimization for {target} {cand}")
                    study = optuna.create_study(
                        directions=["minimize"], sampler=_create_sampler()
                    )
                    study.enqueue_trial(opt_init)
                    study.optimize(objective_peak_angle(opt_init, opt_diff), n_trials=nt)
                    for key, val in study.best_params.items():
                        self.pixia[key] = val

        self.pixia.last_calibrated_keys = list(active_keys)
        self.pixia.last_calibrated_mode = mode
        if self.pixia is not None:
            self.pixia._state = "CALIBRATED"

    def _optimize_lmfit(self, cand, active_keys, mode="pn"):
        """Refines active geometry parameters using Trust Region Reflective (TRF) Non-linear Least Squares."""
        import lmfit
        import scipy.ndimage
        from scipy.optimize import least_squares
        from orochi.pixia.database import standard_diffraction

        step = 2
        orig_mask = getattr(self, "original_mask", None)
        if orig_mask is None:
            orig_mask = self.pixia.mask if getattr(self.pixia, "mask", None) is not None else self.pixia.get_mask()
        mask_sub = (orig_mask != 0)[::step, ::step]
        ibgs_sub = self.pixia.ibgs[::step, ::step].astype(np.float32)
        y_sub, x_sub = np.mgrid[0 : self.pixia["pixy"] : step, 0 : self.pixia["pixx"] : step]
        x_sub = x_sub.astype(np.float32)
        y_sub = y_sub.astype(np.float32)
        psx = float(self.pixia["pix_sizex"])
        psy = float(self.pixia["pix_sizey"])

        ttmin = 1.0
        ttmax = float(self.pixia["ttmax"]) if self.pixia["ttmax"] is not None else 26.0
        ttmax = min(ttmax, 28.0)
        ttstep = 0.005
        x_grid = np.arange(ttmin, ttmax + ttstep, ttstep, dtype=np.float64)

        std_key = str(self.standard or "ni").lower()
        lam_init = float(self.pixia["rambda"])
        try:
            std_data = standard_diffraction.get_data(std_key)
            d_spacing = std_data["d_spacing"]
            rel_int = std_data["relative_intensity"]
            tt_all = 2.0 * np.degrees(np.arcsin(np.clip(lam_init / (2.0 * d_spacing), 0.0, 1.0)))
            valid = (tt_all >= ttmin) & (tt_all <= ttmax) & (rel_int >= 0.8)
            d_val = d_spacing[valid]
            r_val = rel_int[valid]
            tt_val = tt_all[valid]

            unique_d = []
            for cur_d, cur_r, cur_tt in zip(d_val, r_val, tt_val):
                if not unique_d or abs(cur_tt - 2.0 * np.degrees(np.arcsin(np.clip(lam_init / (2.0 * unique_d[-1]), 0.0, 1.0)))) > 0.15:
                    unique_d.append(cur_d)
            d_prom = np.array(unique_d, dtype=np.float64)
            if len(d_prom) < 3:
                prominent = rel_int >= 5.0
                d_prom = d_spacing[prominent]
        except Exception:
            d_prom = np.array([2.035, 1.762, 1.246, 1.063, 1.018, 0.881, 0.809, 0.788, 0.720, 0.678], dtype=np.float64)

        def _calc_target_bragg(rambda_val):
            sin_theta = rambda_val / (2.0 * d_prom)
            v = sin_theta <= 1.0
            tt_calc = 2.0 * np.degrees(np.arcsin(sin_theta[v]))
            in_range = (tt_calc >= ttmin) & (tt_calc <= ttmax)
            return tt_calc[in_range]

        default_bragg = _calc_target_bragg(lam_init)

        def _calc_peak_windows(bragg_angles):
            windows = []
            for idx, pt in enumerate(bragg_angles):
                d_left = pt - bragg_angles[idx - 1] if idx > 0 else 2.0
                d_right = bragg_angles[idx + 1] - pt if idx < len(bragg_angles) - 1 else 2.0
                win = min(0.35, min(d_left, d_right) * 0.45)
                windows.append(win)
            return np.array(windows, dtype=np.float64)

        default_windows = _calc_peak_windows(default_bragg)

        def _get_peaks(x_arr, y_arr, target_peaks, peak_windows=None):
            if peak_windows is None:
                peak_windows = default_windows
            y_smooth = scipy.ndimage.gaussian_filter1d(y_arr, sigma=1.5)
            pks = []
            for pt, w in zip(target_peaks, peak_windows):
                m = (x_arr >= pt - w) & (x_arr <= pt + w)
                if np.count_nonzero(m) < 5:
                    pks.append(np.nan)
                    continue
                sub_x, sub_y = x_arr[m], y_smooth[m]
                idx = np.argmax(sub_y)
                k0 = max(0, idx - 4)
                k1 = min(len(sub_x), idx + 5)
                if k1 - k0 >= 5:
                    bg_line = np.linspace(sub_y[k0], sub_y[k1 - 1], k1 - k0)
                    fit_p = np.polyfit(sub_x[k0:k1], sub_y[k0:k1] - bg_line, 2)
                    if fit_p[0] < 0:
                        pks.append(-fit_p[1] / (2.0 * fit_p[0]))
                        continue
                pks.append(sub_x[idx])
            return np.asarray(pks)

        mode_str = str(mode).lower()
        is_corner_mode = mode_str not in ("pn", "xn-xp,yn-yp", "centered")

        # Distance pre-alignment: if distance is to be optimized, estimate it from observed Bragg rings
        if "distance" in cand:
            try:
                import scipy.signal
                dist_cur = float(np.ravel(self.pixia["distance"])[0])
                cx_cur = float(np.ravel(self.pixia["x0"])[0])
                cy_cur = float(np.ravel(self.pixia["y0"])[0])
                tx_cur = float(np.ravel(self.pixia["tiltx"])[0])
                ty_cur = float(np.ravel(self.pixia["tilty"])[0])

                dx_init = (x_sub - cx_cur) * psx
                dy_init = (y_sub - cy_cur) * psy
                beta_init = np.degrees(np.arctan2(dy_init, dx_init))
                t0_init = np.array([np.sin(np.radians(tx_cur)), np.sin(np.radians(ty_cur)), 1.0], dtype=np.float32) * dist_cur
                bx_init = dx_init - t0_init[0]
                by_init = dy_init - t0_init[1]
                bz_init = -t0_init[2]
                beam_len_init = np.sqrt(bx_init * bx_init + by_init * by_init + bz_init * bz_init)
                t0_len_init = np.linalg.norm(t0_init)
                dot_init = -(bx_init * t0_init[0] + by_init * t0_init[1] + bz_init * t0_init[2])
                cos_tt_init = np.clip(dot_init / (beam_len_init * t0_len_init), -1.0, 1.0)
                tt_deg_init = np.degrees(np.arccos(cos_tt_init))

                if is_corner_mode:
                    if "xn-yp" in mode_str:
                        a1, a2, ad = 180.0, 90.0, 135.0
                    elif "xp-yn" in mode_str:
                        a1, a2, ad = 0.0, -90.0, -45.0
                    elif "xn-yn" in mode_str:
                        a1, a2, ad = 180.0, -90.0, -135.0
                    else:
                        a1, a2, ad = 0.0, 90.0, 45.0
                    m1 = mask_sub & (np.minimum(np.abs(beta_init - a1), 360.0 - np.abs(beta_init - a1)) < 15.0)
                    m2 = mask_sub & (np.minimum(np.abs(beta_init - a2), 360.0 - np.abs(beta_init - a2)) < 15.0)
                    md = mask_sub & (np.minimum(np.abs(beta_init - ad), 360.0 - np.abs(beta_init - ad)) < 15.0)
                    prof_1 = profile_calculation(np.ascontiguousarray(tt_deg_init[m1], dtype=np.float64), np.ascontiguousarray(ibgs_sub[m1], dtype=np.float64), x_grid, False)
                    prof_2 = profile_calculation(np.ascontiguousarray(tt_deg_init[m2], dtype=np.float64), np.ascontiguousarray(ibgs_sub[m2], dtype=np.float64), x_grid, False)
                    prof_d = profile_calculation(np.ascontiguousarray(tt_deg_init[md], dtype=np.float64), np.ascontiguousarray(ibgs_sub[md], dtype=np.float64), x_grid, False)
                    profs = [prof_1, prof_2, prof_d]
                else:
                    r_m = mask_sub & (np.abs(beta_init) < 15.0)
                    l_m = mask_sub & (np.abs(np.abs(beta_init) - 180.0) < 15.0)
                    t_m = mask_sub & (np.abs(beta_init - 90.0) < 15.0)
                    b_m = mask_sub & (np.abs(beta_init + 90.0) < 15.0)
                    prof_r = profile_calculation(np.ascontiguousarray(tt_deg_init[r_m], dtype=np.float64), np.ascontiguousarray(ibgs_sub[r_m], dtype=np.float64), x_grid, False)
                    prof_l = profile_calculation(np.ascontiguousarray(tt_deg_init[l_m], dtype=np.float64), np.ascontiguousarray(ibgs_sub[l_m], dtype=np.float64), x_grid, False)
                    prof_t = profile_calculation(np.ascontiguousarray(tt_deg_init[t_m], dtype=np.float64), np.ascontiguousarray(ibgs_sub[t_m], dtype=np.float64), x_grid, False)
                    prof_b = profile_calculation(np.ascontiguousarray(tt_deg_init[b_m], dtype=np.float64), np.ascontiguousarray(ibgs_sub[b_m], dtype=np.float64), x_grid, False)
                    profs = [prof_r, prof_l, prof_t, prof_b]

                def _find_p0(p):
                    p_s = scipy.ndimage.gaussian_filter1d(p, sigma=2.0)
                    pks, _ = scipy.signal.find_peaks(p_s, prominence=np.max(p_s) * 0.05, distance=int(0.2 / ttstep))
                    return x_grid[pks[0]] if len(pks) > 0 else np.nan

                p0_vals = [_find_p0(p) for p in profs]
                valid_p0 = [v for v in p0_vals if not np.isnan(v)]
                if len(valid_p0) >= 2:
                    avg_p0 = float(np.mean(valid_p0))
                    scale_ratio = float(np.tan(np.radians(avg_p0)) / np.tan(np.radians(default_bragg[0])))
                    if 0.7 <= scale_ratio <= 1.4:
                        d_pre = dist_cur * scale_ratio
                        p_min_d = float(np.ravel(self.pixia.par.get("distance", {}).get("min", d_pre - 20.0))[0])
                        p_max_d = float(np.ravel(self.pixia.par.get("distance", {}).get("max", d_pre + 20.0))[0])
                        d_pre = np.clip(d_pre, p_min_d, p_max_d)
                        self.pixia["distance"] = d_pre
                        print(f"[Calibrator] Pre-aligned distance from {dist_cur:.2f} mm to {d_pre:.3f} mm based on Bragg ring positions.")
            except Exception as e:
                print(f"[Calibrator] Notice: distance pre-alignment skipped ({e})")

        # Set bounds and normalize to z in [0, 1]
        lb_list = []
        ub_list = []
        vec0_list = []
        for k in cand:
            val = float(np.ravel(self.pixia[k])[0])
            p_min_raw = self.pixia.par.get(k, {}).get("min", None)
            p_max_raw = self.pixia.par.get(k, {}).get("max", None)
            p_min = float(np.ravel(p_min_raw)[0]) if p_min_raw is not None else None
            p_max = float(np.ravel(p_max_raw)[0]) if p_max_raw is not None else None

            # If bounds are missing, invalid, or current val is outside bounds, compute natural bounds around val
            if p_min is None or p_max is None or p_min >= p_max or val < p_min or val > p_max:
                if k in ("x0", "y0"):
                    p_min, p_max = val - 5.0, val + 5.0
                elif k in ("tiltx", "tilty"):
                    p_min, p_max = val - 1.0, val + 1.0
                elif k == "distance":
                    p_min, p_max = val * 0.95, val * 1.05
                elif k == "rambda":
                    p_min, p_max = val * 0.97, val * 1.03
                else:
                    p_min, p_max = val - abs(val) * 0.05, val + abs(val) * 0.05
                if k in self.pixia.par:
                    self.pixia.par[k]["min"] = [float(p_min)]
                    self.pixia.par[k]["max"] = [float(p_max)]

            val = float(np.clip(val, p_min, p_max))
            lb_list.append(p_min)
            ub_list.append(p_max)
            vec0_list.append(val)

        lb = np.array(lb_list, dtype=np.float64)
        ub = np.array(ub_list, dtype=np.float64)
        span = ub - lb
        span = np.where(span <= 0, 1.0, span)
        z0 = np.clip((np.array(vec0_list) - lb) / span, 0.0, 1.0)

        width = 15.0

        if is_corner_mode:
            # Multi-arc circularity across the quadrant
            if "xn-yp" in mode_str:
                arc_angles = [180.0, 157.5, 135.0, 112.5, 90.0]
            elif "xp-yn" in mode_str:
                arc_angles = [0.0, -22.5, -45.0, -67.5, -90.0]
            elif "xn-yn" in mode_str:
                arc_angles = [180.0, -157.5, -135.0, -112.5, -90.0]
            else:  # xp-yp
                arc_angles = [0.0, 22.5, 45.0, 67.5, 90.0]

            def residual_corner(z):
                p_vec = lb + z * span
                p_map = dict(zip(cand, p_vec))
                cx = p_map.get("x0", float(self.pixia["x0"]))
                cy = p_map.get("y0", float(self.pixia["y0"]))
                tx = p_map.get("tiltx", float(self.pixia["tiltx"]))
                ty = p_map.get("tilty", float(self.pixia["tilty"]))
                dist = p_map.get("distance", float(self.pixia["distance"]))
                lam = p_map.get("rambda", float(self.pixia["rambda"]))

                t0 = np.array([np.sin(np.radians(tx)), np.sin(np.radians(ty)), 1.0], dtype=np.float32) * dist
                dx = (x_sub - cx) * psx
                dy = (y_sub - cy) * psy
                beta = np.degrees(np.arctan2(dy, dx))
                bx = dx - t0[0]
                by = dy - t0[1]
                bz = -t0[2]
                beam_len = np.sqrt(bx * bx + by * by + bz * bz)
                t0_len = np.linalg.norm(t0)
                dot = -(bx * t0[0] + by * t0[1] + bz * t0[2])
                cos_tt = np.clip(dot / (beam_len * t0_len), -1.0, 1.0)
                tt_deg = np.degrees(np.arccos(cos_tt))

                cur_bragg = _calc_target_bragg(lam) if "rambda" in p_map else default_bragg
                cur_windows = _calc_peak_windows(cur_bragg) if "rambda" in p_map else default_windows

                sector_pks = []
                for ang in arc_angles:
                    diff_ang = np.abs(beta - ang)
                    diff_ang = np.minimum(diff_ang, 360.0 - diff_ang)
                    s_idx = mask_sub & (diff_ang < width)
                    if not np.any(s_idx):
                        sector_pks.append(np.full(len(cur_bragg), np.nan))
                        continue
                    prof = profile_calculation(
                        np.ascontiguousarray(tt_deg[s_idx], dtype=np.float64),
                        np.ascontiguousarray(ibgs_sub[s_idx], dtype=np.float64),
                        x_grid,
                        False,
                    )
                    sector_pks.append(_get_peaks(x_grid, prof, cur_bragg, cur_windows))

                sector_pks = np.array(sector_pks)
                mean_pks = np.nanmean(sector_pks, axis=0)

                res_arc = []
                for s_pk in sector_pks:
                    d = s_pk - mean_pks
                    res_arc.append(np.where(np.isnan(d), 0.5, d) * 3.0)
                res_arc = np.concatenate(res_arc)

                d_bragg = mean_pks - cur_bragg
                res_bragg = np.where(np.isnan(d_bragg), 0.5, d_bragg) * 3.0

                return np.concatenate([res_arc, res_bragg])

            res = least_squares(
                residual_corner,
                z0,
                bounds=(0.0, 1.0),
                method="trf",
                diff_step=0.003,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=150,
            )
        else:
            def residual_pn(z):
                p_vec = lb + z * span
                p_map = dict(zip(cand, p_vec))
                cx = p_map.get("x0", float(self.pixia["x0"]))
                cy = p_map.get("y0", float(self.pixia["y0"]))
                tx = p_map.get("tiltx", float(self.pixia["tiltx"]))
                ty = p_map.get("tilty", float(self.pixia["tilty"]))
                dist = p_map.get("distance", float(self.pixia["distance"]))
                lam = p_map.get("rambda", float(self.pixia["rambda"]))

                t0 = np.array([np.sin(np.radians(tx)), np.sin(np.radians(ty)), 1.0], dtype=np.float32) * dist
                dx = (x_sub - cx) * psx
                dy = (y_sub - cy) * psy
                beta = np.degrees(np.arctan2(dy, dx))
                bx = dx - t0[0]
                by = dy - t0[1]
                bz = -t0[2]
                beam_len = np.sqrt(bx * bx + by * by + bz * bz)
                t0_len = np.linalg.norm(t0)
                dot = -(bx * t0[0] + by * t0[1] + bz * t0[2])
                cos_tt = np.clip(dot / (beam_len * t0_len), -1.0, 1.0)
                tt_deg = np.degrees(np.arccos(cos_tt))

                r_idx = mask_sub & (np.abs(beta) < width)
                l_idx = mask_sub & (np.abs(np.abs(beta) - 180.0) < width)
                t_idx = mask_sub & (np.abs(beta - 90.0) < width)
                b_idx = mask_sub & (np.abs(beta + 90.0) < width)

                cur_bragg = _calc_target_bragg(lam) if "rambda" in p_map else default_bragg
                cur_windows = _calc_peak_windows(cur_bragg) if "rambda" in p_map else default_windows

                i_r = profile_calculation(np.ascontiguousarray(tt_deg[r_idx], dtype=np.float64), np.ascontiguousarray(ibgs_sub[r_idx], dtype=np.float64), x_grid, False)
                i_l = profile_calculation(np.ascontiguousarray(tt_deg[l_idx], dtype=np.float64), np.ascontiguousarray(ibgs_sub[l_idx], dtype=np.float64), x_grid, False)
                i_t = profile_calculation(np.ascontiguousarray(tt_deg[t_idx], dtype=np.float64), np.ascontiguousarray(ibgs_sub[t_idx], dtype=np.float64), x_grid, False)
                i_b = profile_calculation(np.ascontiguousarray(tt_deg[b_idx], dtype=np.float64), np.ascontiguousarray(ibgs_sub[b_idx], dtype=np.float64), x_grid, False)

                pk_r = _get_peaks(x_grid, i_r, cur_bragg, cur_windows)
                pk_l = _get_peaks(x_grid, i_l, cur_bragg, cur_windows)
                pk_t = _get_peaks(x_grid, i_t, cur_bragg, cur_windows)
                pk_b = _get_peaks(x_grid, i_b, cur_bragg, cur_windows)

                diff_h = pk_r - pk_l
                res_h = np.where(np.isnan(diff_h), 0.5, diff_h) * 3.0

                diff_v = pk_t - pk_b
                res_v = np.where(np.isnan(diff_v), 0.5, diff_v) * 3.0

                pk_avg = np.nanmean([pk_r, pk_l, pk_t, pk_b], axis=0)
                diff_bragg = pk_avg - cur_bragg
                res_bragg = np.where(np.isnan(diff_bragg), 0.5, diff_bragg) * 3.0

                tot = np.concatenate([res_h, res_v, res_bragg])
                return tot

            res = least_squares(
                residual_pn,
                z0,
                bounds=(0.0, 1.0),
                method="trf",
                diff_step=0.003,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=150,
            )

        final_p = lb + res.x * span
        for k, val in zip(cand, final_p):
            self.pixia[k] = float(val)

        res_obj = lmfit.minimizer.MinimizerResult()
        res_obj.nfev = res.nfev
        res_obj.chisqr = float(2.0 * res.cost)
        res_obj.params = lmfit.Parameters()
        for k, val in zip(cand, final_p):
            res_obj.params.add(k, value=float(val), min=float(self.pixia.par[k]["min"][0]), max=float(self.pixia.par[k]["max"][0]))

        print(f"[OK] LeastSquares (TRF) geometry calibration finished (nfev={res.nfev}, cost={res.cost:.3e}) for {cand}")
        return res_obj

    def get_directional_profiles(self, width: float = 10.0, standard: str | None = None, mode: str = "pn"):
        """Extract directional 1D profiles (+X, -X, +Y, -Y, diagonals) and theoretical Bragg peaks.

        Parameters
        ----------
        width : float, optional
            Angular sector half-width in degrees (default: 10.0).
        standard : str, optional
            Calibrant standard name (e.g. "ni", "ceo2"). Defaults to self.standard.
        mode : str, optional
            Calibration geometry mode (default: "pn" for opposing centered beam;
            or "xp-yp", "xn-yp", "xp-yn", "xn-yn" for corner beam geometries).

        Returns
        -------
        dict
            Dictionary containing 1D arrays and metadata for geometry validation:
            - "mode": Selected geometry mode.
            - "tt_deg": 2theta array (deg).
            - "i_pos_x": Right (+X, beta = 0° ± width).
            - "i_neg_x": Left (-X, beta = 180° ± width).
            - "i_pos_y": Top (+Y, beta = 90° ± width).
            - "i_neg_y": Bottom (-Y, beta = -90° ± width).
            - "i_diag_tr": Top-Right diagonal (+X/+Y, beta = 45° ± width).
            - "i_diag_tl": Top-Left diagonal (-X/+Y, beta = 135° ± width).
            - "i_diag_br": Bottom-Right diagonal (+X/-Y, beta = -45° ± width).
            - "i_diag_bl": Bottom-Left diagonal (-X/-Y, beta = -135° ± width).
            - "radius_mm": Corresponding radial detector distance in mm.
            - "diff_h": Horizontal asymmetry score (+X vs -X).
            - "diff_v": Vertical asymmetry score (+Y vs -Y).
            - "diff_xp_yp": Orthogonal score (+X vs +Y).
            - "diff_xn_yp": Orthogonal score (-X vs +Y).
            - "diff_xp_yn": Orthogonal score (+X vs -Y).
            - "diff_xn_yn": Orthogonal score (-X vs -Y).
            - "std_name": Standard calibrant name.
            - "std_bragg_tt": Theoretical Bragg 2theta peaks (deg).
            - "std_bragg_radius": Theoretical Bragg radii on detector (mm).
            - "std_bragg_intensity": Relative Bragg intensities.
            - "std_bragg_d": Bragg d-spacings (A).
        """
        if self.pixia is None:
            raise ValueError("Pixia instance is required.")

        if not isinstance(getattr(self.pixia, "ibgs", None), np.ndarray):
            if not isinstance(getattr(self.pixia, "sample", None), np.ndarray):
                self.pixia.read()
            self.pixia.bg_subtraction()
        if getattr(self.pixia, "mask", None) is None or not isinstance(self.pixia.mask, np.ndarray):
            self.pixia.mask = self.pixia.get_mask()

        y, x = np.mgrid[0 : self.pixia["pixy"], 0 : self.pixia["pixx"]]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        mask = (getattr(self.pixia, "mask", None) != 0) if getattr(self.pixia, "mask", None) is not None else np.ones_like(x, dtype=bool)
        ibgs = self.pixia.ibgs.astype(np.float32)

        dist = float(self.pixia["distance"])
        psx = float(self.pixia["pix_sizex"])
        psy = float(self.pixia["pix_sizey"])

        ttmin = float(self.pixia["ttmin"]) if self.pixia["ttmin"] is not None else 1.0
        ttmax = float(self.pixia["ttmax"]) if self.pixia["ttmax"] is not None else 30.0
        ttstep = float(self.pixia["ttstep"]) if self.pixia["ttstep"] is not None else 0.01
        x_grid = np.arange(ttmin, ttmax + ttstep, ttstep, dtype=np.float64)

        t0 = np.array(
            [np.sin(np.radians(self.pixia["tiltx"])), np.sin(np.radians(self.pixia["tilty"])), 1.0],
            dtype=np.float32,
        ) * dist
        dx = (x - float(self.pixia["x0"])) * psx
        dy = (y - float(self.pixia["y0"])) * psy

        beta = np.degrees(np.arctan2(dy, dx))

        bx = dx - t0[0]
        by = dy - t0[1]
        bz = -t0[2]
        beam_len = np.sqrt(bx * bx + by * by + bz * bz)
        t0_len = np.linalg.norm(t0)
        dot = -(bx * t0[0] + by * t0[1] + bz * t0[2])
        cos_tt = np.clip(dot / (beam_len * t0_len), -1.0, 1.0)
        tt_deg = np.degrees(np.arccos(cos_tt))

        def _calc_profile(cond):
            if not np.any(cond):
                return np.zeros_like(x_grid)
            return profile_calculation(
                np.ascontiguousarray(tt_deg[cond], dtype=np.float64),
                np.ascontiguousarray(ibgs[cond], dtype=np.float64),
                x_grid,
                False,
            )

        r_idx = mask & (np.abs(beta) < width)
        l_idx = mask & (np.abs(np.abs(beta) - 180.0) < width)
        t_idx = mask & (np.abs(beta - 90.0) < width)
        b_idx = mask & (np.abs(beta + 90.0) < width)

        tr_idx = mask & (np.abs(beta - 45.0) < width)
        tl_idx = mask & (np.abs(beta - 135.0) < width)
        br_idx = mask & (np.abs(beta + 45.0) < width)
        bl_idx = mask & (np.abs(beta + 135.0) < width)

        i_r = _calc_profile(r_idx)
        i_l = _calc_profile(l_idx)
        i_t = _calc_profile(t_idx)
        i_b = _calc_profile(b_idx)

        i_diag_tr = _calc_profile(tr_idx)
        i_diag_tl = _calc_profile(tl_idx)
        i_diag_br = _calc_profile(br_idx)
        i_diag_bl = _calc_profile(bl_idx)

        diff_h = float(np.sum((i_r - i_l) ** 2) / (np.sum(i_r**2) + 1e-9))
        diff_v = float(np.sum((i_t - i_b) ** 2) / (np.sum(i_t**2) + 1e-9))

        diff_xp_yp = float(np.sum((i_r - i_t) ** 2) / (np.sum(i_r**2) + np.sum(i_t**2) + 1e-9))
        diff_xn_yp = float(np.sum((i_l - i_t) ** 2) / (np.sum(i_l**2) + np.sum(i_t**2) + 1e-9))
        diff_xp_yn = float(np.sum((i_r - i_b) ** 2) / (np.sum(i_r**2) + np.sum(i_b**2) + 1e-9))
        diff_xn_yn = float(np.sum((i_l - i_b) ** 2) / (np.sum(i_l**2) + np.sum(i_b**2) + 1e-9))

        radius_mm = dist * np.tan(np.radians(x_grid))

        std_key = str(standard or self.standard or "ni").lower()
        std_tt = []
        std_rad = []
        std_int = []
        std_d = []
        try:
            from orochi.pixia.database import standard_diffraction
            std_data = standard_diffraction.get_data(std_key)
            d_spacing = std_data["d_spacing"]
            rel_int = std_data["relative_intensity"]
            rambda = float(self.pixia["rambda"])
            sin_theta = rambda / (2.0 * d_spacing)
            valid = sin_theta <= 1.0
            d_valid = d_spacing[valid]
            rel_int_valid = rel_int[valid]
            tt_calc = 2.0 * np.degrees(np.arcsin(sin_theta[valid]))
            in_range = (tt_calc >= ttmin) & (tt_calc <= ttmax)
            std_tt = tt_calc[in_range]
            std_rad = dist * np.tan(np.radians(std_tt))
            std_int = rel_int_valid[in_range]
            std_d = d_valid[in_range]
        except Exception as e:
            if hasattr(self.pixia, "_logger"):
                self.pixia._logger.warning(f"Could not load standard peaks for {std_key}: {e}")

        return {
            "mode": mode,
            "tt_deg": x_grid,
            "i_pos_x": i_r,
            "i_neg_x": i_l,
            "i_pos_y": i_t,
            "i_neg_y": i_b,
            "i_diag_tr": i_diag_tr,
            "i_diag_tl": i_diag_tl,
            "i_diag_br": i_diag_br,
            "i_diag_bl": i_diag_bl,
            "radius_mm": radius_mm,
            "diff_h": diff_h,
            "diff_v": diff_v,
            "diff_xp_yp": diff_xp_yp,
            "diff_xn_yp": diff_xn_yp,
            "diff_xp_yn": diff_xp_yn,
            "diff_xn_yn": diff_xn_yn,
            "std_name": std_key,
            "std_bragg_tt": np.asarray(std_tt),
            "std_bragg_radius": np.asarray(std_rad),
            "std_bragg_intensity": np.asarray(std_int),
            "std_bragg_d": np.asarray(std_d),
        }

    def set_mask(self):
        if getattr(self.pixia, "mask", None) is None:
            self.pixia.mask = self.pixia.get_mask()
        filtered1 = ndim.gaussian_filter(self.pixia.sample, sigma=1)
        filtered2 = ndim.gaussian_filter(self.pixia.sample, sigma=10)
        mask = np.where(filtered1 > filtered2 * 1.1, 1, 0)
        self.original_mask = self.pixia.mask_image
        self.pixia.mask_image = self.pixia.mask_image * mask
        self.pixia.mask = self.pixia.get_mask()

    def reset_mask(self):
        self.pixia.mask_image = self.original_mask
        self.pixia.mask = self.pixia.get_mask()

    def sparse_data(self):
        self.pixia.t0 = (
            np.array(
                [
                    np.sin(np.radians(self.pixia["tiltx"])),
                    np.sin(np.radians(self.pixia["tilty"])),
                    1.0,
                ],
                dtype=self.pixia.digit,
            )
            * self.pixia["distance"]
        )
        self.pixia.xyz_coordinates, self.pixia.normals = self.pixia.get_coordinates()
        if getattr(self.pixia, "mask", None) is None:
            self.pixia.mask = self.pixia.get_mask()
        r = np.where(self.pixia.mask != 0)

        coordinates = self.pixia.xyz_coordinates[r]
        beta = self.pixia.get_beta_radian(coordinates)

        import copy
        betamax = copy.deepcopy(self.pixia["betamax"])
        betamin = copy.deepcopy(self.pixia["betamin"])
        if isinstance(betamax, np.ndarray):
            betamax = betamax.item() if betamax.ndim == 0 else betamax.tolist()
        if isinstance(betamin, np.ndarray):
            betamin = betamin.item() if betamin.ndim == 0 else betamin.tolist()

        self.pixia["betamax"] = (np.arange(0, 360, 10) + 1).tolist()
        self.pixia["betamin"] = (np.arange(0, 360, 10) - 1).tolist()

        rb = np.where(self.pixia.get_beta_range(beta) != 0)
        self.pixia["betamin"] = betamin
        self.pixia["betamax"] = betamax

        coordinates = coordinates[rb]
        tt2d = self.pixia.get_tt_radian(coordinates)

        beta2d = beta[rb]
        polarization = self.pixia.get_polar_corrction(tt2d, beta2d)
        if not isinstance(getattr(self.pixia, "ibgs", None), np.ndarray):
            if not isinstance(getattr(self.pixia, "sample", None), np.ndarray):
                self.pixia.read()
            self.pixia.bg_subtraction()
        i2d = self.pixia.ibgs[r][rb] / polarization

        x_new = np.arange(
            self.pixia["ttmin"],
            self.pixia["ttmax"] + self.pixia["ttstep"],
            self.pixia["ttstep"],
        )

        tt2d_deg = np.degrees(tt2d)
        i1d = profile_calculation(
            np.ascontiguousarray(tt2d_deg.flatten(), dtype=np.float64),
            np.ascontiguousarray(i2d.flatten(), dtype=np.float64),
            np.ascontiguousarray(x_new, dtype=np.float64),
            False,
        )
        return x_new, i1d

    def center_analysis(self):
        self.pixia.t0 = (
            np.array(
                [
                    np.sin(np.radians(self.pixia["tiltx"])),
                    np.sin(np.radians(self.pixia["tilty"])),
                    1.0,
                ],
                dtype=self.pixia.digit,
            )
            * self.pixia["distance"]
        )
        self.pixia.xyz_coordinates, self.pixia.normals = self.pixia.get_coordinates()
        if getattr(self.pixia, "mask", None) is None:
            self.pixia.mask = self.pixia.get_mask()
        r = np.where(self.pixia.mask != 0)

        coordinates = self.pixia.xyz_coordinates[r]
        beta = self.pixia.get_beta_radian(coordinates)

        angles = np.arange(0, 180, 45.0)
        d_list = []
        import copy
        betamax = copy.deepcopy(self.pixia["betamax"])
        betamin = copy.deepcopy(self.pixia["betamin"])
        if isinstance(betamax, np.ndarray):
            betamax = betamax.item() if betamax.ndim == 0 else betamax.tolist()
        if isinstance(betamin, np.ndarray):
            betamin = betamin.item() if betamin.ndim == 0 else betamin.tolist()

        for center in angles:
            self.pixia["betamax"] = [center + 1.0]
            self.pixia["betamin"] = [center - 1.0]
            pos = np.where(self.pixia.get_beta_range(beta) != 0)

            self.pixia["betamax"] = [center - 180.0 + 1.0]
            self.pixia["betamin"] = [center - 180.0 - 1.0]
            neg = np.where(self.pixia.get_beta_range(beta) != 0)

            tt_pos = np.degrees(self.pixia.get_tt_radian(coordinates[pos]))
            tt_neg = np.degrees(self.pixia.get_tt_radian(coordinates[neg]))
            i_pos = self.pixia.ibgs[r][pos]
            i_neg = self.pixia.ibgs[r][neg]

            tt_max = np.amin([np.amax(tt_pos), np.amax(tt_neg)])
            tt_min = np.amax([np.amin(tt_pos), np.amin(tt_neg)])
            tt = np.linspace(tt_min, tt_max, 1000)
            i1d_pos = profile_calculation(
                np.ascontiguousarray(tt_pos.flatten(), dtype=np.float64),
                np.ascontiguousarray(i_pos.flatten(), dtype=np.float64),
                np.ascontiguousarray(tt, dtype=np.float64),
                False,
            )
            i1d_neg = profile_calculation(
                np.ascontiguousarray(tt_neg.flatten(), dtype=np.float64),
                np.ascontiguousarray(i_neg.flatten(), dtype=np.float64),
                np.ascontiguousarray(tt, dtype=np.float64),
                False,
            )

            d_list.append(np.average(np.abs(i1d_pos - i1d_neg)))

        self.pixia["betamax"] = betamax
        self.pixia["betamin"] = betamin

        return d_list

    def peak_analysis(self, tt1d, i1d):
        """Fits individual standard Bragg peaks with a Gaussian to evaluate peak positions and widths."""

        def peak_list_from_standard_data():
            standard_data = standard_diffraction.get_data(str(self.standard).lower())
            d_spacing = standard_data["d_spacing"]
            relative_intensity = standard_data["relative_intensity"]

            q_std = 2.0 * np.pi / d_spacing
            sorted_idx = np.argsort(q_std)
            q_std = q_std[sorted_idx]
            i_std = relative_intensity[sorted_idx]

            use = np.where((q_std > self.q_min) & (q_std < self.q_max))
            q_std = q_std[use]
            i_std = i_std[use]

            use = np.where(
                (np.abs(q_std - np.append(q_std[1:], 0)) > self.peak_width / 2.0)
                & (np.abs(q_std - np.append(0, q_std[:-1])) > self.peak_width / 2.0)
            )
            return q_std[use], i_std[use]

        def peak_fitting(q, i):
            def gaus_func(trial):
                a = trial.suggest_float("a", 0.0, float(top))
                b = trial.suggest_float("b", float(q[0]), float(q[-1]))
                c = trial.suggest_float("c", float(width * 0.001), float(width))
                d = trial.suggest_float("d", 0.0, float(ave))
                score = np.average(np.square(i - (a * np.exp(-((q - b) / c) ** 2) + d)))
                return float(score)

            width = q[-1] - q[0]
            top = np.amax(i)
            ave = np.average(i)

            optuna.logging.disable_default_handler()
            study = optuna.create_study(
                directions=["minimize"], sampler=optuna.samplers.NSGAIISampler()
            )
            study.enqueue_trial(
                {
                    "a": float(top),
                    "b": float((q[-1] + q[0]) / 2.0),
                    "c": float(width / 2.0),
                    "d": float((i[-1] + i[0]) / 2.0),
                }
            )
            study.optimize(gaus_func, n_trials=30)
            optuna.logging.enable_default_handler()

            return study.best_params

        q_std, i_std = peak_list_from_standard_data()

        i = i1d / np.amax(i1d)
        q = self.pixia.twotheta_to_q(tt1d)
        w = self.peak_width / 2.0

        peak_intensity = []
        peak_angle = []
        peak_width = []

        for idx, q_std_n in enumerate(q_std):
            r = np.where((q_std_n - w < q) & (q_std_n + w > q))
            if r[0].size == 0:
                continue
            q_n = q[r]
            i_n = i[r]
            sorted_idx = np.argsort(q_n)
            q_n = q_n[sorted_idx]
            i_n = i_n[sorted_idx]

            best_params = peak_fitting(q_n, i_n)
            peak_intensity.append(best_params["a"] / i_std[idx])
            peak_angle.append(best_params["b"] - q_std_n)
            peak_width.append(best_params["c"])

        return peak_intensity, peak_angle, peak_width

    def auto_beamstop_mask(self, search_box: int = 100, threshold_factor: float = 0.15) -> int:
        """Detect direct beamstop shadow circle automatically around nominal beam center.

        Parameters
        ----------
        search_box : int, optional
            Radial search window radius in pixels (default: 100).
        threshold_factor : float, optional
            Fraction of the outer diffraction level where shadow boundary is defined (default: 0.15).

        Returns
        -------
        int
            Detected beamstop radius in pixels.
        """
        if self.pixia is None:
            raise ValueError("Pixia instance is required for auto_beamstop_mask.")
        if not isinstance(self.pixia.sample, np.ndarray):
            self.pixia.read()

        x0 = int(round(float(self.pixia["x0"])))
        y0 = int(round(float(self.pixia["y0"])))
        sample = self.pixia.sample

        y_grid, x_grid = np.ogrid[: self.pixia["pixy"], : self.pixia["pixx"]]
        r_dist = np.sqrt((x_grid - x0) ** 2 + (y_grid - y0) ** 2)

        r_bins = np.arange(0, search_box, 1)
        r_indices = np.digitize(r_dist.ravel(), r_bins) - 1

        valid_flat = (r_dist < search_box).ravel()
        r_sub = r_indices[valid_flat]
        i_sub = sample.ravel()[valid_flat]

        bin_medians = [
            np.median(i_sub[r_sub == b]) if np.any(r_sub == b) else 0 for b in range(len(r_bins))
        ]
        max_level = np.percentile(bin_medians[len(bin_medians) // 2 :], 90)

        stop_r = 30
        for b, val in enumerate(bin_medians):
            if val > max_level * threshold_factor and b > 10:
                stop_r = b + 3  # Add 3 px safety margin
                break

        self.pixia["beamstop_circle"] = stop_r
        self.pixia.mask = self.pixia.get_mask()
        return stop_r

    def align_center_tilt(
        self, width: float = 15.0, max_iter: int = 30, xatol: float = 0.02, fatol: float = 1e-4
    ) -> dict:
        """Align beam center (x0, y0) and detector tilt angles (tiltx, tilty) using opposing sector symmetry.

        Compares Right (beta ~ 0 deg) vs Left (beta ~ 180 deg) and Top (beta ~ 90 deg) vs Bottom (beta ~ -90 deg).

        Parameters
        ----------
        width : float, optional
            Angular sector half-width in degrees (default: 15.0).
        max_iter : int, optional
            Maximum Nelder-Mead iterations (default: 30).
        xatol : float, optional
            Absolute tolerance in parameter coordinates (default: 0.02).
        fatol : float, optional
            Absolute tolerance in opposing mismatch objective function (default: 1e-4).

        Returns
        -------
        dict
            Dictionary of optimized parameters and cost improvement.
        """
        if self.pixia is None:
            raise ValueError("Pixia instance is required for align_center_tilt.")
        if not isinstance(self.pixia.sample, np.ndarray):
            self.pixia.read()
        self.pixia.bg_subtraction()

        y, x = np.mgrid[0 : self.pixia["pixy"], 0 : self.pixia["pixx"]]
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        mask = self.pixia.mask != 0
        ibgs = self.pixia.ibgs.astype(np.float32)

        dist = float(self.pixia["distance"])
        psx = float(self.pixia["pix_sizex"])
        psy = float(self.pixia["pix_sizey"])
        x_grid = np.arange(3.0, 25.0, 0.02, dtype=np.float64)

        def fast_cost(params):
            cx, cy, tx, ty = params
            t0 = (
                np.array(
                    [np.sin(np.radians(tx)), np.sin(np.radians(ty)), 1.0], dtype=np.float32
                )
                * dist
            )
            dx = (x - cx) * psx
            dy = (y - cy) * psy

            beta = np.degrees(np.arctan2(dy, dx))

            bx = dx - t0[0]
            by = dy - t0[1]
            bz = -t0[2]
            beam_len = np.sqrt(bx * bx + by * by + bz * bz)
            t0_len = np.linalg.norm(t0)
            dot = -(bx * t0[0] + by * t0[1] + bz * t0[2])
            cos_tt = np.clip(dot / (beam_len * t0_len), -1.0, 1.0)
            tt_deg = np.degrees(np.arccos(cos_tt))

            r_idx = mask & (np.abs(beta) < width)
            i_r = profile_calculation(
                np.ascontiguousarray(tt_deg[r_idx], dtype=np.float64),
                np.ascontiguousarray(ibgs[r_idx], dtype=np.float64),
                x_grid,
                False,
            )

            l_idx = mask & (np.abs(np.abs(beta) - 180.0) < width)
            i_l = profile_calculation(
                np.ascontiguousarray(tt_deg[l_idx], dtype=np.float64),
                np.ascontiguousarray(ibgs[l_idx], dtype=np.float64),
                x_grid,
                False,
            )

            t_idx = mask & (np.abs(beta - 90.0) < width)
            i_t = profile_calculation(
                np.ascontiguousarray(tt_deg[t_idx], dtype=np.float64),
                np.ascontiguousarray(ibgs[t_idx], dtype=np.float64),
                x_grid,
                False,
            )

            b_idx = mask & (np.abs(beta + 90.0) < width)
            i_b = profile_calculation(
                np.ascontiguousarray(tt_deg[b_idx], dtype=np.float64),
                np.ascontiguousarray(ibgs[b_idx], dtype=np.float64),
                x_grid,
                False,
            )

            diff_h = np.sum((i_r - i_l) ** 2) / (np.sum(i_r**2) + 1e-9)
            diff_v = np.sum((i_t - i_b) ** 2) / (np.sum(i_t**2) + 1e-9)
            return float(diff_h + diff_v)

        init_p = [
            float(self.pixia["x0"]),
            float(self.pixia["y0"]),
            float(self.pixia["tiltx"]),
            float(self.pixia["tilty"]),
        ]
        init_cost = fast_cost(init_p)

        res = minimize(
            fast_cost,
            init_p,
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": xatol, "fatol": fatol},
        )

        opt_cx, opt_cy, opt_tx, opt_ty = res.x
        self.pixia["x0"] = float(opt_cx)
        self.pixia["y0"] = float(opt_cy)
        self.pixia["tiltx"] = float(opt_tx)
        self.pixia["tilty"] = float(opt_ty)

        return {
            "x0": float(opt_cx),
            "y0": float(opt_cy),
            "tiltx": float(opt_tx),
            "tilty": float(opt_ty),
            "initial_cost": float(init_cost),
            "final_cost": float(res.fun),
            "iterations": int(res.nit),
        }

    def tune_polarization(self, width: float = 15.0) -> float:
        """Calibrates synchrotron polarization factor p by matching horizontal intensity to vertical reference.

        Parameters
        ----------
        width : float, optional
            Angular sector half-width in degrees (default: 15.0).

        Returns
        -------
        float
            Optimal polarization factor p.
        """
        if self.pixia is None:
            raise ValueError("Pixia instance is required for tune_polarization.")

        # Vertical reference slice (P = 1.0)
        self.pixia["polar"] = 1.0
        self.pixia.set_azimuthal_mode("vertical", width=width)
        self.pixia.run(dummy=False)
        tt_v = self.pixia.tt1d.copy()
        i_v = self.pixia.i1d.copy()

        def polar_cost(p_val):
            p = float(np.ravel(p_val)[0])
            self.pixia["polar"] = p
            self.pixia.set_azimuthal_mode("horizontal", width=width)
            self.pixia.run(dummy=False)
            i_h = self.pixia.i1d.copy()

            valid = (tt_v >= 5.0) & (tt_v <= 25.0) & (i_v > 500)
            ratio = i_h[valid] / np.maximum(i_v[valid], 1.0)
            return float(np.mean((ratio - 1.0) ** 2))

        res = minimize(polar_cost, [0.99], bounds=[(0.85, 1.0)], method="L-BFGS-B")
        opt_polar = float(res.x[0])
        self.pixia["polar"] = opt_polar
        return opt_polar

    def calibrate_distance_wavelength_with_pdf(
        self,
        cif_file: str | None = None,
        standard: str = "ni",
        target_lattice: float = 3.5238,
        qmax: float = 20.0,
        fit_r_min: float = 2.2,
        fit_r_max: float = 20.0,
        work_dir: str | None = None,
    ) -> dict:
        """Calibrate sample-to-detector distance L0 and wavelength lambda by refining standard PDF against certified lattice parameter.

        Parameters
        ----------
        cif_file : str, optional
            Path to reference CIF file. If None, bundled standard CIF is loaded.
        standard : str, optional
            Standard calibrant material name (default: "ni").
        target_lattice : float, optional
            Certified target cubic lattice parameter a_0 in Angstrom (default: 3.5238).
        qmax : float, optional
            Maximum Q for Fourier transformation in Angstrom^-1 (default: 20.0).
        fit_r_min : float, optional
            Minimum r for real-space least-squares refinement (default: 2.2).
        fit_r_max : float, optional
            Maximum r for real-space least-squares refinement (default: 20.0).
        work_dir : str, optional
            Working directory for intermediate PDF files. If None, uses temp directory.

        Returns
        -------
        dict
            Calibration results including fitted lattice, scale discrepancy ratio, calibrated distance, and fit Rw.
        """
        import os
        import tempfile
        from orochi.phase import Phase
        from orochi.pdfget import PDFget
        from orochi.pdfsim import Simulator
        from orochi.fit import LeastSquaresOptimizer

        if work_dir is None:
            work_dir = tempfile.gettempdir()
        os.makedirs(work_dir, exist_ok=True)

        if cif_file is None:
            db_cif = os.path.join(os.path.dirname(__file__), "database", f"{standard.lower()}.cif")
            if os.path.exists(db_cif):
                cif_file = db_cif
            else:
                raise FileNotFoundError(f"Standard CIF file for '{standard}' not found at {db_cif}")

        # Run 1D reduction with full azimuthal coverage
        self.pixia.set_azimuthal_mode("full")
        self.pixia.run(dummy=True)

        phase = Phase()
        phase.read_file(cif_file)
        phase["shape_type"] = "analytical"
        phase["sphere"] = [1000.0, 1000.0, 1000.0]
        phase.set_lattice_constraints()
        phase()

        pdfget = PDFget()
        pdfget.from_pixia(self.pixia)
        pdfget.set_phase(phase)
        pdfget["source"] = "synchrotron"
        pdfget["qmin"] = 1.44
        pdfget["qmax"] = qmax
        pdfget["fitmin"] = 10.0
        pdfget["fitmax"] = qmax - 0.5
        pdfget["padding"] = 5
        pdfget.run()

        gr_path = os.path.join(work_dir, f"{standard}_calib_fit.gr")
        pdfget.save_file(gr_path)

        sim = Simulator()
        sim.phase = phase
        sim["rambda"] = float(self.pixia["rambda"])
        sim["qmin"] = 1.44
        sim["qmax"] = qmax
        sim["rmin"] = fit_r_min
        sim["rmax"] = fit_r_max
        sim["rstep"] = 0.01
        sim["qdepend"] = True
        sim()

        phase["lattice", "min"] = [target_lattice * 0.95] * 3 + [89.0] * 3
        phase["lattice", "max"] = [target_lattice * 1.05] * 3 + [91.0] * 3
        phase["lattice", "optimize"] = [True, False, False, False, False, False]
        phase.set_lattice_constraints()
        phase["adp", "min"] = [0.0005]
        phase["adp", "max"] = [0.03]
        phase["adp", "optimize"] = [True]
        phase()

        sim["scale"] = 0.55
        sim["scale", "min"] = 0.1
        sim["scale", "max"] = 5.0
        sim["scale", "optimize"] = True
        sim["qdamp"] = 0.024
        sim["qdamp", "min"] = 0.001
        sim["qdamp", "max"] = 0.1
        sim["qdamp", "optimize"] = True
        sim["qbroad"] = 0.014
        sim["qbroad", "min"] = 0.0001
        sim["qbroad", "max"] = 0.1
        sim["qbroad", "optimize"] = True
        sim["delta2"] = 2.4
        sim["delta2", "min"] = 0.0
        sim["delta2", "max"] = 10.0
        sim["delta2", "optimize"] = True

        opt = LeastSquaresOptimizer()
        opt.work_dir = work_dir
        opt.study_name = f"{standard}_lattice_calib"
        opt.simulators = [sim]
        opt["fit_min"] = fit_r_min
        opt["fit_max"] = fit_r_max
        opt.read_file(gr_path)
        opt()

        fitted_a = float(phase["lattice"][0])
        scale_ratio = target_lattice / fitted_a
        orig_dist = float(self.pixia["distance"])
        calibrated_distance = orig_dist * scale_ratio
        self.pixia["distance"] = calibrated_distance

        return {
            "fitted_a": fitted_a,
            "target_lattice": target_lattice,
            "scale_ratio": scale_ratio,
            "calibrated_distance": calibrated_distance,
            "Rw": opt.Rw * 100.0,
            "adp": float(phase["adp"][0]),
        }

    def auto_pipeline(
        self,
        cif_file: str | None = None,
        standard: str = "ni",
        target_lattice: float = 3.5238,
        qmax: float = 20.0,
        work_dir: str | None = None,
    ) -> dict:
        """Executes the complete multi-stage automated 2D scattering calibration pipeline.

        Pipeline Stages:
        1. Auto-detection of beamstop shadow mask.
        2. Opposing sector symmetry alignment for beam center and detector tilts.
        3. Polarization tuning using vertical reference sectors.
        4. Absolute distance and wavelength calibration via reference standard PDF refinement.

        Returns
        -------
        dict
            Comprehensive calibration report.
        """
        report = {}
        # Stage 1: Auto beamstop mask
        stop_r = self.auto_beamstop_mask()
        report["beamstop_radius"] = stop_r

        # Stage 2: Align center & tilt
        center_tilt = self.align_center_tilt()
        report["center_tilt"] = center_tilt

        # Stage 3: Polarization tuning
        polar = self.tune_polarization()
        report["polarization"] = polar

        # Stage 4: Distance & wavelength calibration with PDF
        dist_report = self.calibrate_distance_wavelength_with_pdf(
            cif_file=cif_file,
            standard=standard,
            target_lattice=target_lattice,
            qmax=qmax,
            work_dir=work_dir,
        )
        report["distance_calibration"] = dist_report
        return report


# Backward-compatible lowercase alias
calibrator = Calibrator

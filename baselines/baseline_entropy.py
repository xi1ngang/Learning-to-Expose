#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy-Weighted Image-Gradient baseline
Kim et al., ICRA 2018  |  Updated 2025-06-11
"""

# --------------------------------------------------------------------------- #
# 0. Imports & settings                                                       #
# --------------------------------------------------------------------------- #
import os
import warnings
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

# skimage is optional – use if present, else fall back to SciPy
try:
    from skimage.filters.rank import entropy as _entropy_rank
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
from scipy.ndimage import generic_filter

# Silence noisy but harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
np.seterr(over="raise", divide="raise", invalid="raise")     # fail fast


# --------------------------------------------------------------------------- #
# 1. reward & smoothness  (identical to your PID baseline)                    #
# --------------------------------------------------------------------------- #
def calculate_reward(image: np.ndarray) -> float:
    img = image.astype(np.float32) / 255.0
    flat = img.flatten()

    def qmean(x: np.ndarray, lo: float, hi: float) -> float:
        a, b = np.quantile(x, lo), np.quantile(x, hi)
        m = (x >= a) & (x <= b)
        return float(x[m].mean()) if m.any() else 0.0

    eps = 5 / 255
    r1 = 1 if 50/255 - eps <= flat.mean() <= 50/255 + eps else 0
    r2 = 1 if 5/255  <= qmean(flat, 0.0, 0.2) <= 100/255 else 0
    r3 = 1 if 10/255 <= qmean(flat, 0.2, 0.4) <= 100/255 else 0
    r4 = 1 if 12.5/255 <= qmean(flat, 0.4, 0.8) <= 80/255 else 0
    r5 = 1 if 100/255 <= qmean(flat, 0.8, 1.0) <= 250/255 else 0
    return 0.1*r1 + 0.3*(r2+r3+r5) + 0.1*r4


def compute_smoothness(actions: List[int]) -> float:
    a = np.asarray(actions, np.float32)
    if a.size < 2:
        return 1.0
    num = abs(float(a[0]) - float(a[-1]))
    den = np.abs(np.diff(a)).sum()
    return float(num / den) if den > 0 else 1.0


# --------------------------------------------------------------------------- #
# 2. Entropy & EWG metric                                                     #
# --------------------------------------------------------------------------- #
def _square_footprint(k: int):
    """Return a k×k footprint regardless of skimage version."""
    if not _HAS_SKIMAGE:
        return np.ones((k, k), dtype=np.uint8)
    try:                                      # skimage < 0.25
        from skimage.morphology import square
        return square(k)
    except (ImportError, TypeError):          # skimage ≥ 0.25
        from skimage.morphology import footprint_rectangle
        return footprint_rectangle(k, k)


def _local_entropy(gray: np.ndarray, k: int = 9) -> np.ndarray:
    """Per-pixel entropy in a k×k window, normalised to [0, 1]."""
    g = gray.astype(np.uint8)
    if _HAS_SKIMAGE:
        ent = _entropy_rank(g, _square_footprint(k)).astype(np.float32)
        return ent / 8.0                               # 8 = log₂(256)
    else:                                              # fallback, slower
        def _ent(vec):
            cnt = np.bincount(vec, minlength=256)
            p = cnt / cnt.sum()
            p = p[p > 0]
            return -(p*np.log2(p)).sum()
        ent = generic_filter(g, _ent, size=k)
        return (ent / np.log2(256)).astype(np.float32)


def ewg_score(
    gray: np.ndarray,
    alpha: float = 10.0,
    tau: float = 2.0,
    h_thres: float = 0.05,
    eps: float = 1e-8,
) -> float:
    """Kim et al. EWG metric – always finite."""
    H = _local_entropy(gray)
    sigma = max(float(H.std()), 1e-3)

    # Gaussian weight  (correct minus sign)
    try:
        w = np.exp(-((H - H.mean()) ** 2) / (2 * sigma ** 2))
    except FloatingPointError:
        w = np.ones_like(H, np.float32)
    W = w / max(w.sum(), 1.0)

    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    mean_grad = float(grad.mean())

    pi = 2 / (1 + np.exp(-alpha * H + tau)) - 1
    M = (H < h_thres).astype(np.float32)

    g_i = W * grad + pi * M * W * mean_grad
    return float(g_i.sum() + eps)


# --------------------------------------------------------------------------- #
# 3. Bayesian-optimising exposure controller                                  #
# --------------------------------------------------------------------------- #
class EWGBoController:
    """Gaussian-process BO with MAXVAR / MAXMI acquisition."""

    def __init__(
        self,
        exposure_range: Tuple[int, int] = (1, 111),
        max_iter: int = 60,
        var_tol: float = 0.5,
        min_samples: int = 3,
        mode: str = "MAXVAR",
    ):
        self.grid = np.arange(exposure_range[0], exposure_range[1] + 1)[:, None]
        self.max_iter, self.var_tol, self.min_samples = max_iter, var_tol, min_samples
        self.mode = mode.upper()

        ker = 5.0 * RBF(length_scale=15.0, length_scale_bounds=(5.0, 60.0))
        ker += WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e-1))
        self.gp = GaussianProcessRegressor(ker, alpha=1e-6, normalize_y=True)

    # ---- helper ----------------------------------------------------------- #
    @staticmethod
    def _load_gray(img_path: str) -> np.ndarray:
        return np.array(Image.open(img_path).convert("L"))

    def _score(self, folder: str, exp_idx: int) -> float:
        path = os.path.join(folder, f"{exp_idx}.png")
        return ewg_score(self._load_gray(path))

    # ---- main search ------------------------------------------------------ #
    def search(self, folder: str, init_exp: int) -> Tuple[int, int, List[int]]:
        X, y = [[init_exp]], [self._score(folder, init_exp)]
        traj = [init_exp]                # track exposure trajectory
        eta = 0.0                        # MAXMI exploration term

        for _ in range(self.max_iter):
            self.gp.fit(np.asarray(X), np.asarray(y))
            mu, sigma = self.gp.predict(self.grid, return_std=True)

            # pick candidate ------------------------------------------------
            order = np.argsort(-sigma if self.mode == "MAXVAR" else
                               -(mu + np.sqrt(0.5*(np.sqrt(sigma**2+eta)-np.sqrt(eta)))))
            next_exp = None
            for idx in order:
                candidate = int(self.grid[idx, 0])
                if candidate not in {x[0] for x in X}:
                    next_exp = candidate
                    sig_val = sigma[idx]
                    break
            if next_exp is None:                    # explored whole grid
                break

            # early-stop rule ----------------------------------------------
            if len(X) >= self.min_samples and sig_val < self.var_tol:
                best = int(self.grid[np.argmax(mu), 0])
                return best, len(traj), traj

            # sample & continue --------------------------------------------
            X.append([next_exp])
            y.append(self._score(folder, next_exp))
            traj.append(next_exp)
            eta += sig_val ** 2                         # for MAXMI

        best = int(X[np.argmax(y)][0])
        return best, len(traj), traj


# --------------------------------------------------------------------------- #
# 4. Wrapper for your evaluation harness                                      #
# --------------------------------------------------------------------------- #
def adjust_exposure_entropy(
    folder: str,
    init_exp: int,
    controller: EWGBoController,
):
    best_exp, steps, traj = controller.search(folder, init_exp)
    img = Image.open(os.path.join(folder, f"{best_exp}.png")).convert("L")
    reward = calculate_reward(np.array(img))
    smooth = compute_smoothness(traj)
    return reward, smooth, steps


# --------------------------------------------------------------------------- #
# 5. Driver (mirrors your PID script)                                         #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":

    BASE = (
        "/Users/xingang/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/"
        "Learning-to-Expose/ExpoSweep"
    )
    ROOT = os.path.join(BASE, "evaluation", "indoor")    # adapt if needed
    folders = sorted(os.listdir(ROOT))

    ctrl = EWGBoController(mode="MAXVAR")                 # or "MAXMI"

    rewards, smooths, steps, fails = [], [], [], 0
    for sub in folders:
        path = os.path.join(ROOT, sub)
        for init in (4, 51, 89):
            try:
                r, s, st = adjust_exposure_entropy(path, init, ctrl)
            except Exception as ex:
                print(f"[FAIL] {sub}/{init}: {ex}")
                fails += 1
                continue
            if r == 0:
                fails += 1
                continue
            rewards.append(r)
            smooths.append(s)
            steps.append(st)

    trials = 3 * len(folders)
    print("\nTrials           :", trials)
    print("Not converged    :", fails)
    print("Convergence rate :", 1 - fails / trials)
    print("Avg reward       :", np.mean(rewards))
    print("Avg smoothness   :", np.mean(smooths))
    print("Avg BO steps     :", np.mean(steps))

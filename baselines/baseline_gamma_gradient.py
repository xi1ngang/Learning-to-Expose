import os, cv2
import numpy as np
from PIL import Image


def calculate_reward(image: np.ndarray) -> float:
    """
    Calculate a reward based on image brightness distribution using NumPy.

    Args:
        image (np.ndarray): Input image as a NumPy array with pixel values in [0, 255].

    Returns:
        float: Reward value.
    """

    # Normalize image to [0, 1]
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    flat_image = image.flatten()

    def average(img: np.ndarray) -> float:
        return np.mean(img)

    def quantile_mean(img: np.ndarray, lower_q: float, upper_q: float) -> float:
        lower_bound = np.quantile(img, lower_q)
        upper_bound = np.quantile(img, upper_q)
        mask = (img >= lower_bound) & (img <= upper_bound)
        return np.mean(img[mask]) if np.any(mask) else 0.0

    epsilon = 5 / 255

    # Conditions
    r1 = 1 if (50/255 - epsilon) <= average(flat_image) <= (50/255 + epsilon) else 0
    r2 = 1 if (5/255 <= quantile_mean(flat_image, 0, 0.2) <= 100/255) else 0
    r3 = 1 if (10/255 <= quantile_mean(flat_image, 0.2, 0.4) <= 100/255) else 0
    r4 = 1 if (12.5/255 <= quantile_mean(flat_image, 0.4, 0.8) <= 80/255) else 0
    r5 = 1 if (100/255 <= quantile_mean(flat_image, 0.8, 1.0) <= 250/255) else 0

    # Weights
    weight1 = 0.1
    weight2 = 0.3
    weight3 = 0.3
    weight4 = 0.1
    weight5 = 0.3

    # Final reward
    stat_reward = (
        (r1 * weight1)
        + (r2 * weight2)
        + (r3 * weight3)
        + (r4 * weight4)
        + (r5 * weight5)
    )

    return stat_reward


def compute_smoothness(actions: list | np.ndarray) -> float:
    """
    Compute the smoothness reward of an action trajectory.

    Args:
        actions: A 1D list or NumPy array of scalar actions (length ≥ 2)

    Returns:
        A float in [0, 1] representing the smoothness reward
    """
    actions = np.array(actions, dtype=np.float32)
    
    if len(actions) < 2:
        return 1.0  # Consider a single-element sequence perfectly smooth

    numerator = abs(actions[0] - actions[-1])
    denominator = np.sum(np.abs(np.diff(actions)))

    return float(numerator / denominator) if denominator != 0 else 1.0


# ----------------- 1. Gradient information metric ----------------- #
def gradient_information(img_gray, delta=0.02, lamb=1e3):
    """Eq.(1) on a [0,1] gray image (no per-frame normalisation)."""
    img_f = img_gray.astype(np.float32) / 255.0          # [0,1]
    sobelx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)                 # already ~[0,1]

    mask = mag >= delta
    mapped = np.zeros_like(mag, dtype=np.float32)
    mapped[mask] = np.log(lamb * (mag[mask] - delta) + 1.0)
    mapped /= np.log(lamb * (1 - delta) + 1.0)
    return float(mapped.sum())

# ----------------- 2. Gamma correction helper ----------------- #
def gamma_correct(img, gamma):
    # img float32 in [0,1]
    return np.power(img, gamma, dtype=np.float32)

# ----------------- 3. Controller ----------------- #
class GradientExposureController:
    def __init__(self, kp=0.2, d=0.75,
                 anchors=(1/1.9, 1/1.5, 1/1.2, 1.0, 1.2, 1.5, 1.9),
                 min_exp=1, max_exp=111):
        self.kp, self.d = kp, d
        self.anchors = anchors
        self.min_exp, self.max_exp = min_exp, max_exp

    def _estimate_gamma(self, img_gray):
        metrics = []
        img_f = img_gray.astype(np.float32) / 255.0
        for g in self.anchors:
            g_img = np.power(img_f, g, dtype=np.float32)
            metrics.append(gradient_information((g_img*255).astype(np.uint8)))

        metrics = np.array(metrics)
        best = np.where(metrics == metrics.max())[0]
        # choose γ whose value is nearest to 1.0
        chosen = best[np.argmin(np.abs(np.array(self.anchors)[best] - 1.0))]
        return self.anchors[int(chosen)]

    def _nonlinear_update(self, gamma_hat, exposure):
        alpha = 1.0 if gamma_hat < 1.0 else 0.5
        R = self.d * np.tan((2 - gamma_hat) * np.arctan(1 / self.d) - np.arctan(1 / self.d)) + 1
        new_exp = (1 + alpha * self.kp * (R - 1)) * exposure
        return int(np.clip(new_exp, self.min_exp, self.max_exp))

    def update(self, img_gray, exposure):
        gamma_hat = self._estimate_gamma(img_gray)
        next_exp = self._nonlinear_update(gamma_hat, exposure)
        return next_exp, gamma_hat

# ----------------- 4. Evaluation loop (mirrors your PID baseline) ----------------- #
def adjust_exposure_gradient(folder, initial_exposure,
                             controller=GradientExposureController(),
                             gamma_tol=0.08, max_iter=60):
    exp = initial_exposure
    trajectory = []

    for _ in range(max_iter):
        trajectory.append(exp)
        img_path = os.path.join(folder, f"{exp}.png")
        img = Image.open(img_path).convert("L")  # grayscale
        img_gray = np.array(img)

        new_exp, gamma_hat = controller.update(img_gray, exp)

        if abs(gamma_hat - 1.0) < gamma_tol:
            reward = calculate_reward(img_gray)   # ← reuse your existing function
            smooth = compute_smoothness(trajectory)
            return reward, smooth, len(trajectory)-1

        exp = new_exp

    return 0.0, 0.0, 0  # did not converge

# ----------------- 5. Example driver ----------------- #
if __name__ == "__main__":
    base = "/Users/xingang/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Learning-to-Expose/ExpoSweep"
    folder = os.path.join(base, "evaluation", "indoor")
    subsets = os.listdir(folder)

    rewards, smooths, speeds, fails = [], [], [], 0
    for sub in subsets:
        path = os.path.join(folder, sub)
        for init_exp in [4, 51, 89]:
            r, s, sp = adjust_exposure_gradient(path, init_exp)
            if r == 0: fails += 1; continue
            rewards.append(r); smooths.append(s); speeds.append(sp)

    print("Avg reward:", np.mean(rewards))
    print("Avg smoothness:", np.mean(smooths))
    print("Avg iterations:", np.mean(speeds))
    print("Convergence rate:", 1 - fails/(3*len(subsets)))

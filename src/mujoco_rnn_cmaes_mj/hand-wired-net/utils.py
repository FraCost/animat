import os
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.colors import LinearSegmentedColormap


def get_root_path():
    root_path = os.path.abspath(os.path.dirname(__file__))
    while root_path != os.path.dirname(root_path):
        if os.path.exists(os.path.join(root_path, ".git")):
            break
        root_path = os.path.dirname(root_path)
    return root_path


def exponential_kernel(tau, time):
    """Generates an exponential kernel parameterized by its mean"""
    lambda_ = 1 / tau
    kernel = lambda_ * np.exp(-lambda_ * time)
    return kernel / kernel.sum()


def truncated_exponential(mu, a, b, size=1):
    """Sample from a truncated exponential distribution using inverse CDF method."""
    lambda_ = 1 / mu
    U = np.random.uniform(0, 1, size)
    exp_a, exp_b = np.exp(-lambda_ * a), np.exp(-lambda_ * b)
    return np.array(-np.log((1 - U) * (exp_a - exp_b) + exp_b) / lambda_)


def sample_entropy(samples, base=2):
    """Compute entropy directly from a vector of samples."""
    _, counts = np.unique(samples, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs) / np.log(base))


def beta_from_mean(mu, nu=5, num_samples=1):
    alpha = mu * nu
    beta_ = (1 - mu) * nu
    return beta.rvs(alpha, beta_, size=num_samples)


def logistic(x, k=1.0, c=0.0):
    return 1 / (1 + np.exp(-k * (x - c)))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def softpus(x):
    return np.log(1 + np.exp(x))


def saturating_exponential(x, k=1.0):
    return 1 - np.exp(-k * x)


def xavier_init(n_in, n_out):
    stddev = np.sqrt(1 / (n_in + n_out))
    return np.random.randn(n_out, n_in) * stddev


def he_init(n_in, n_out):
    """He (Kaiming) initialization for ReLU weights."""
    stddev = np.sqrt(2 / n_in)
    return np.random.randn(n_out, n_in) * stddev


def l1_norm(x):
    return np.sum(np.abs(x))


def l2_norm(x):
    return np.sqrt(np.sum(x**2))


def normalize01(x, xmin, xmax, default=0.5):
    valid = xmax > xmin
    xnorm = np.full_like(x, default)
    xnorm[valid] = (x[valid] - xmin[valid]) / (xmax[valid] - xmin[valid])
    return xnorm


def zscore(x, xmean, xstd, default=0):
    valid = xstd > 0
    xnorm = np.full_like(x, default)
    xnorm[valid] = (x[valid] - xmean[valid]) / xstd[valid]
    return xnorm


def action_entropy(action, base=2):
    action = np.clip(action, 1e-10, 1)  # Avoid log(0)
    action_pdf = action / action.sum()
    return -np.sum(action_pdf * np.log(action_pdf) / np.log(base))


def analyze_damped_oscillation(t, x, settle_threshold=0.02, plot=False):
    """
    Parameters
    ----------
    t : array_like
        Time array (s)
    x : array_like
        Amplitude/displacement array (>=0)
    settle_threshold : float, optional
        Amplitude fraction for "settling" (default = 0.02 = 2%)
    plot : bool, optional
        If True, plots the signal with peaks and exponential envelope.
    """

    # --- Step 1: Convert to arrays ---
    t = np.asarray(t)
    x = np.asarray(x)

    # --- Step 2: Estimate steady-state offset (final value) ---
    # Take the mean of the last 10% of samples as steady-state value
    N_tail = max(10, len(x) // 10)
    x_offset = np.mean(x[-N_tail:])

    # --- Step 3: Work with the decaying part (relative to offset) ---
    x_centered = x - x_offset
    if np.all(x_centered <= 0):
        raise ValueError("Signal has no positive oscillation above the steady-state offset.")

    # --- Step 4: Find peaks in the centered signal ---
    peaks, _ = find_peaks(x_centered)

    t_peaks = t[peaks]
    A_peaks = x_centered[peaks]
    A_max = A_peaks[0]

    # --- Step 5: Logarithmic decrement and damping ratio ---
    delta = np.mean(np.log(A_peaks[:-1] / A_peaks[1:]))  # average over peak pairs
    zeta = delta / np.sqrt((2 * np.pi)**2 + delta**2)

    # --- Step 6: Damped frequency and natural frequency ---
    Td = np.mean(np.diff(t_peaks))       # average period between peaks
    omega_d = 2 * np.pi / Td
    omega_0 = omega_d / np.sqrt(1 - zeta**2)

    # --- Step 7: Settling time and amplitude ---
    Ts = 4 / (zeta * omega_0)                                # time to reach ~2% band
    A_settle = A_max * np.exp(-zeta * omega_0 * Ts)           # amplitude at Ts
    settle_band = x_offset + A_settle                         # absolute amplitude at Ts
    
    print('final angle', A_settle + x_offset)

    # --- Step 8: Optional plot ---
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(t, x, lw=1.5, label="Signal")
        plt.plot(t_peaks, x[peaks], "ro", label="Peaks")

        # Exponential decay envelope
        env = x_offset + A_max * np.exp(-zeta * omega_0 * t)
        plt.plot(t, env, "k--", alpha=0.7, label="Exponential envelope")

        # Settling band around steady-state
        plt.axhline(settle_band, color="g", ls="--", lw=1, label="Settling band (2%)")
        plt.axhline(x_offset, color="gray", ls=":", lw=1, label="Steady-state offset")

        plt.xlabel("Time (s)", fontsize=18)
        plt.ylabel("Joint angle (deg)", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(-5, 65)
        plt.tight_layout()
        plt.show()

    # --- Step 9: Return results ---
    return {
        'damping_ratio': zeta,
        'logarithmic_decrement': delta,
        'settling_time': Ts,
        'settling_amplitude': A_settle + x_offset,
        'peak_amplitude': A_max + x_offset,
        'steady_state_offset': x_offset
    }



def color_gradient(min, max, N):
    cmap = LinearSegmentedColormap.from_list("blue_gradient", [min, max], N=N)
    return [cmap(i/(N-1)) for i in range(N)]
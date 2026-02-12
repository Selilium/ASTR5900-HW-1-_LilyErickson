
import numpy as np
import matplotlib.pyplot as plt

# --- Try SciPy for off-the-shelf cubic spline ---
try:
    from scipy.interpolate import CubicSpline
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# -----------------------------
# True function
# y = sin((pi/2)x) + x/2
# -----------------------------
def f(x):
    return np.sin((np.pi / 2.0) * x) + x / 2.0


# -----------------------------
# Custom piecewise linear interpolation
# Assumes x_data sorted, no extrapolation
# -----------------------------
def linear_interp(x_new, x_data, y_data):
    x_new = np.asarray(x_new)
    y_new = np.empty_like(x_new, dtype=float)

    xmin, xmax = x_data[0], x_data[-1]
    if np.any(x_new < xmin) or np.any(x_new > xmax):
        raise ValueError("x_new contains values outside the data domain; no extrapolation allowed.")

    # For each x_new, find the interval index i such that x_data[i] <= x < x_data[i+1]
    idx = np.searchsorted(x_data, x_new, side="right") - 1
    idx = np.clip(idx, 0, len(x_data) - 2)

    x0 = x_data[idx]
    x1 = x_data[idx + 1]
    y0 = y_data[idx]
    y1 = y_data[idx + 1]

    t = (x_new - x0) / (x1 - x0)
    y_new = y0 + t * (y1 - y0)
    return y_new


def main():
    # -----------------------------
    # Dataset: integers 0..10
    # -----------------------------
    x_data = np.arange(0, 11, 1, dtype=float)
    y_data = f(x_data)

    # 10× higher resolution across same domain:
    # original has 11 points -> make 101 points (10 intervals * 10 + 1)
    x_hi = np.linspace(0, 10, 101)
    y_true_hi = f(x_hi)

    # Linear (custom)
    y_lin_hi = linear_interp(x_hi, x_data, y_data)

    # Cubic spline (off-the-shelf)
    if not SCIPY_OK:
        raise ImportError(
            "SciPy not found. Install it (pip install scipy) or run this in an environment that has SciPy."
        )
    cs = CubicSpline(x_data, y_data, bc_type="natural")
    y_spl_hi = cs(x_hi)

    # -----------------------------
    # (a) Plot dataset + both interpolants
    # -----------------------------
    plt.figure()
    plt.plot(x_data, y_data, "o", label="Data (integers 0..10)")
    plt.plot(x_hi, y_lin_hi, "-", label="Custom linear interp (10×)")
    plt.plot(x_hi, y_spl_hi, "--", label="Cubic spline (SciPy) (10×)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(r"Problem 3(a): $y=\sin\left(\frac{\pi}{2}x\right)+\frac{x}{2}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig3a_interp_compare.png", dpi=300)

    # -----------------------------
    # (b) Relative error vs true
    # rel err = (y_interp - y_true)/y_true
    # Avoid division by ~0: mask where |y_true| is tiny
    # -----------------------------
    eps = 1e-12
    mask = np.abs(y_true_hi) > eps

    rel_lin = np.full_like(y_true_hi, np.nan, dtype=float)
    rel_spl = np.full_like(y_true_hi, np.nan, dtype=float)
    rel_lin[mask] = (y_lin_hi[mask] - y_true_hi[mask]) / y_true_hi[mask]
    rel_spl[mask] = (y_spl_hi[mask] - y_true_hi[mask]) / y_true_hi[mask]

    plt.figure()
    plt.plot(x_hi, rel_lin, "-", label="Linear relative error")
    plt.plot(x_hi, rel_spl, "--", label="Spline relative error")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("x")
    plt.ylabel(r"Relative error $(y_{\mathrm{interp}}-y_{\mathrm{true}})/y_{\mathrm{true}}$")
    plt.title("Problem 3(b): Relative error vs true function")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig3b_relative_error.png", dpi=300)

    print("Saved: fig3a_interp_compare.png")
    print("Saved: fig3b_relative_error.png")


if __name__ == "__main__":
    main()

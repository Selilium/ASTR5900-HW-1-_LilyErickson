
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def main():

    # ----- Load dataset -----
    data = np.loadtxt("HW01_data.txt", skiprows=1)
    x = data[:, 0]
    y = data[:, 1]

    # Sort in case data is not ordered
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # ----- Create 10× higher resolution grid -----
    N = len(x)
    x_dense = np.linspace(x.min(), x.max(), 10 * N)

    # ----- Cubic spline interpolation -----
    spline = CubicSpline(x, y)
    y_spline = spline(x_dense)

    # ----- Plot -----
    plt.figure()
    plt.plot(x, y, "o", label="Original dataset")
    plt.plot(x_dense, y_spline, "-", label="Cubic spline interpolation")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Problem 2(b): Cubic Spline Interpolation")
    plt.legend()

    plt.tight_layout()
    plt.savefig("fig2b_spline.png", dpi=300)
    print("Saved fig2b_spline.png")

    plt.show()


if __name__ == "__main__":
    main()

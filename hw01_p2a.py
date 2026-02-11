
import numpy as np
import matplotlib.pyplot as plt

def read_xy(path: str):
    data = np.loadtxt(path, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def linear_interp(x_data, y_data, xq, *, extrapolation="forbid"):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    xq = np.asarray(xq, dtype=float)

    order = np.argsort(x_data)
    x = x_data[order]
    y = y_data[order]

    xmin, xmax = x[0], x[-1]

    if extrapolation == "forbid":
        if np.any((xq < xmin) | (xq > xmax)):
            raise ValueError("Extrapolation requested but extrapolation='forbid'.")
    elif extrapolation == "constant":
        xq = np.clip(xq, xmin, xmax)
    else:
        raise ValueError("Unknown extrapolation mode.")

    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, len(x) - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]

    a = (y1 - y0) / (x1 - x0)
    yq = y0 + a * (xq - x0)
    return yq


def main():
    x, y = read_xy("HW01_data.txt")

    N = len(x)
    x_dense = np.linspace(np.min(x), np.max(x), 10 * N)
    y_dense = linear_interp(x, y, x_dense, extrapolation="forbid")

    plt.figure()
    plt.plot(x_dense, y_dense, ".", label="linear interpolation (10× points)")
    plt.plot(x, y, "o", label="original dataset")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("HW01 Data: Piecewise Linear Interpolation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2a_linear.png", dpi=200)

    print("Saved figure: fig2a_linear.png")
    plt.show()


if __name__ == "__main__":
    main()
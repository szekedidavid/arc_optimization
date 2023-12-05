import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

C1 = 0.0214 * 10 ** -6
C2 = 0.134 * 10 ** -6
g = 9.80665
mu0 = 1.25663706 * 10 ** -6
um = 20 * 10 ** 3  # todo is this a reasonable value?


def get_I_sp(arr: np.ndarray, invert: bool):
    h, w, c, R, L = arr

    m_bit = C1 * h / w * E0 / R

    Lp = mu0 / np.pi * (3 / 2 + np.log(h / (w + c))) * um

    I_bit = E0 / R * (Lp / 2 + C2 / np.sqrt(w) * h / w * (E0 / L) ** (1 / 4))

    I_sp = I_bit / (m_bit * g)

    if invert:
        return 1 / I_sp
    else:
        return I_sp


def optimize_thruster():
    guess = np.array([h0, w0, c0, R0, L0])
    bounds = [(hmin, hmax), (wmin, wmax), (cmin, cmax), (Rmin, Rmax), (Lmin, Lmax)]

    result = sp.optimize.minimize(get_I_sp, guess, (True,), bounds=bounds)

    return result


def investigate_correlation(var: str):
    x_arr = np.linspace(globals()[var + "min"], globals()[var + "max"], 100)
    Isp_arr = np.zeros(100)
    x = 0

    arr_dict = {"h": h0, "w": w0, "c": c0, "R": R0, "L": L0}

    for i in range(np.size(x_arr)):
        arr_dict[var] = x_arr[i]
        arr = np.fromiter(arr_dict.values(), dtype=float)
        Isp_arr[i] = get_I_sp(arr, False)

    fig, ax = plt.subplots()
    ax.plot(x_arr, Isp_arr)
    ax.set_ylim(0, 30000)
    plt.show()
    print(Isp_arr)


f = 5
P = 20

E0 = P / f

hmin, hmax = 0.01, 0.03
wmin, wmax = 0.01, 0.03
cmin, cmax = 0.001, 0.003
Rmin, Rmax = 22 * 10 ** -3, 50 * 10 ** -3
Lmin, Lmax = 45 * 10 ** -9, 170 * 10 ** -9  # todo actually figure out inductance

h0 = (hmin + hmax) / 2
w0 = (wmin + wmax) / 2
c0 = (cmin + cmax) / 2
R0 = (Rmin + Rmax) / 2
L0 = (Lmin + Lmax) / 2

#  investigate_correlation("w")
x = optimize_thruster()
print(x.success, x.x)

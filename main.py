import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

C1 = 0.0214
C2 = 0.234  # todo 0.134 or 0.234??
g = 9.80665
mu0 = 1.25663706
um = 20e3
Rm = 2


def get_results(arr: np.ndarray, opt: bool):
    h, w, c, R, L, E = arr

    m_bit = C1 * (h / w) * (E / R)

    Lp = mu0 / np.pi * (3 / 2 + np.log(h / (w + c)))

    I_EM = E / R * (Lp / 2)
    I_GD = E / R * (C2 / np.sqrt(w)) * (h / w) * (E / L) ** (1 / 4)

    I_bit = I_EM + I_GD

    I_sp = I_bit * 1e-6 / (m_bit * 1e-9 * g)

    fm = I_EM * 1e-6 / (um * m_bit * 1e-9)

    fI = I_EM / I_bit

    eta = (I_bit * 1e-6) ** 2 / (2 * m_bit * 1e-9 * E)

    R_p = Lp * 10 ** -6 * um / Rm

    R_c = R - R_p

    EpA = E / (h * w)

    if opt:
        return 1 / I_sp
    else:
        return I_EM, I_GD, I_bit, m_bit, fI, fm, I_sp, eta, R_c, R_p, EpA


def investigate_correlation(var: str):
    x_arr = np.linspace(globals()[var + "min"], globals()[var + "max"], 100)
    I_EM = np.zeros(100)
    I_GD = np.zeros(100)
    I_bit = np.zeros(100)
    m_bit = np.zeros(100)
    fI = np.zeros(100)
    fm = np.zeros(100)
    I_sp = np.zeros(100)
    eta = np.zeros(100)

    arr_dict = {"h": h0, "w": w0, "c": c0, "R": R0, "L": L0, "E": E0}

    for i in range(np.size(x_arr)):
        arr_dict[var] = x_arr[i]
        arr = np.fromiter(arr_dict.values(), dtype=float)
        I_EM[i], I_GD[i], I_bit[i], m_bit[i], fI[i], fm[i], I_sp[i], eta[i], _, _ = get_results(arr, False)

    fig, axs = plt.subplots(4, 2, figsize=(10, 12))

    fig.suptitle(f"Varying {var}")

    # Plot each array on a separate subplot
    axs[0, 0].plot(x_arr, I_EM, color='b')
    axs[0, 0].set_title('$I_{EM}$')

    axs[0, 1].plot(x_arr, I_GD, color='g')
    axs[0, 1].set_title('$I_{GD}$')

    axs[1, 0].plot(x_arr, I_bit, color='r')
    axs[1, 0].set_title('$I_{bit}$')

    axs[1, 1].plot(x_arr, m_bit, color='c')
    axs[1, 1].set_title('$m_{bit}$')

    axs[2, 0].plot(x_arr, fI, color='m')
    axs[2, 0].set_title('$f_I$')

    axs[2, 1].plot(x_arr, fm, color='y')
    axs[2, 1].set_title('$f_m$')

    axs[3, 0].plot(x_arr, I_sp, color='k')
    axs[3, 0].set_title('$I_{sp}$')

    axs[3, 1].plot(x_arr, eta, color='orange')
    axs[3, 1].set_title('$\eta$')

    # Adjust layout to prevent overlapping titles
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


def optimize_thruster():
    guess = np.array([h0, w0, c0, R0, L0, E0])
    bounds = [(hmin, hmax), (wmin, wmax), (cmin, cmax), (Rmin, Rmax), (Lmin, Lmax), (Emin, Emax)]

    result = sp.optimize.minimize(get_results, guess, (True,), bounds=bounds)

    return result


wmin, wmax = 0.5, 3
cmin, cmax = 0.5, 1
Rmin, Rmax = 22e-3, 55e-3
Lmin, Lmax = 45, 170
Emin, Emax = 20, 20

h0, w0, c0, R0, L0, E0 = 4, 2, 0.5, 35e-3, 100, 20

# investigate_correlation("h")
opt_arr = optimize_thruster().x

opt_res = get_results(opt_arr, False)

print(
    f" h = {opt_arr[0]} \n w = {opt_arr[1]} \n c = {opt_arr[2]} \n R = {opt_arr[3]} \n L = {opt_arr[4]} \n E = {opt_arr[5]}")
print(
    f" I_EM = {opt_res[0]}\n I_GD = {opt_res[1]}\n I_bit = {opt_res[2]}\n m_bit = {opt_res[3]}\n fI = {opt_res[4]}\n fm = {opt_res[5]}\n I_sp = {opt_res[6]}\n eta = {opt_res[7]}\n R_c = {opt_res[8]}\n R_p = {opt_res[9]}\n EpA = {opt_res[10]}")

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def print_results(input, res):
    print(
        f"E0 = {input[5]} (discharge energy [J])\n"
        f"h = {input[0]} (height [cm])\n"
        f"w = {input[1]} (width [cm])\n"
        f"c = {input[2]} (electrode thickness [cm])\n"
        f"R = {input[3]} (total effective resistance [Ohm])\n"
        f"L = {input[4]} (total effective inductance [nH])\n"
        f"I_EM = {res[0]} (electromagnetic impulse bit [microNs])\n"
        f"I_GD = {res[1]} (gas dynamic impulse bit [microNs])\n"
        f"I_bit = {res[2]} (total impulse bit [microNs])\n"
        f"m_bit = {res[3]} (total mass bit [microg])\n"
        f"fI = {res[4]} (electromagnetic impulse fraction [-])\n"
        f"fm = {res[5]} (electromagnetic mass fraction [-])\n"
        f"I_sp = {res[6]} (specific impulse [s])\n"
        f"eta = {res[7]} (efficiency [-])\n"
        f"R_c = {res[8]} (circuit resistance [Ohm])\n"
        f"R_p = {res[9]} (plasma resistance [Ohm])\n"
        f"E0A = {res[10]} (energy to propellant surface area ratio [J/cm^2])"
    )


class ThrusterOptimizer:
    def __init__(self, params, consts):
        self.params = params
        self.consts = consts

        self.guess = np.array([self.params[k][2] for k in self.params])

    def get_results(self, arr: np.ndarray, opt: bool):
        h, w, c, R, L, E = arr
        C1, C2, g, mu0, um, Rm = self.consts.values()

        m_bit = C1 * (h / w) * (E / R) * 1000
        Lp = mu0 / np.pi * (3 / 2 + np.log((h / w) / (1 + (c / w))))
        I_EM = E / R * (Lp / 2) * 1000
        I_GD = (E / R * (C2 / np.sqrt(w)) * (h / w) * (E / L) ** (1 / 4)) * 1000
        I_bit = I_EM + I_GD
        I_sp = I_bit * 1e-9 / (m_bit * 1e-12 * g)
        fm = I_EM * 1e-6 / (um * m_bit * 1e-9)
        fI = I_EM / I_bit
        eta = (I_bit * 1e-9) ** 2 / (2 * m_bit * 1e-12 * E)

        R_p = (Lp * 10 ** -6 * um / Rm) * 1000
        R_c = R - R_p
        EpA = E / (h * w)

        if opt:
            return 1 / I_sp
        else:
            return I_EM, I_GD, I_bit, m_bit, fI, fm, I_sp, eta, R_c, R_p, EpA

    def investigate_correlation(self, var: str):
        x_arr = np.linspace(self.params[var][0], self.params[var][1], 100)
        results = {key: np.zeros(100) for key in ['I_EM', 'I_GD', 'I_bit', 'm_bit', 'fI', 'fm', 'I_sp', 'eta']}
        arr_dict = {key: self.params[key][2] for key in self.params}

        for i in range(np.size(x_arr)):
            arr_dict[var] = x_arr[i]
            arr = np.fromiter(arr_dict.values(), dtype=float)
            results['I_EM'][i], results['I_GD'][i], results['I_bit'][i], results['m_bit'][i], results['fI'][i], \
                results['fm'][i], results['I_sp'][i], results['eta'][i], _, _, _ = self.get_results(arr, False)

        fig, axs = plt.subplots(4, 2, figsize=(10, 12))
        fig.suptitle(f"Varying {var}")

        axs[0, 0].plot(x_arr, results['I_EM'], color='b')
        axs[0, 0].set_title('$I_{EM}$')

        axs[0, 1].plot(x_arr, results['I_GD'], color='g')
        axs[0, 1].set_title('$I_{GD}$')

        axs[1, 0].plot(x_arr, results['I_bit'], color='r')
        axs[1, 0].set_title('$I_{bit}$')

        axs[1, 1].plot(x_arr, results['m_bit'], color='c')
        axs[1, 1].set_title('$m_{bit}$')

        axs[2, 0].plot(x_arr, results['fI'], color='m')
        axs[2, 0].set_title('$f_I$')

        axs[2, 1].plot(x_arr, results['fm'], color='y')
        axs[2, 1].set_title('$f_m$')

        axs[3, 0].plot(x_arr, results['I_sp'], color='k')
        axs[3, 0].set_title('$I_{sp}$')

        axs[3, 1].plot(x_arr, results['eta'], color='orange')
        axs[3, 1].set_title('$\eta$')

        plt.tight_layout(rect=(0, 0, 1, 1))
        plt.show()

    def optimize_thruster(self):
        bounds = [(self.params[k][0], self.params[k][1]) for k in self.params]
        result = sp.optimize.minimize(self.get_results, self.guess, (True,), bounds=bounds)
        return result

    def print_optimized_results(self):
        opt_arr = self.optimize_thruster().x
        opt_res = self.get_results(opt_arr, False)

        print_results(opt_arr, opt_res)

    def print_guess_results(self):
        guess_res = self.get_results(self.guess, False)

        print_results(self.guess, guess_res)


parameters = {
    "h": [1, 4, 4],
    "w": [0.5, 3, 1],
    "c": [0.5, 1, 0.5],
    "R": [22e-3, 55e-3, 35e-3],
    "L": [45, 170, 100],
    "E": [5, 60, 20]
}

constants = {
    "C1": 0.0214,
    "C2": 0.134,
    "g": 9.80665,
    "mu0": 1.25663706,
    "um": 20e3,
    "Rm": 2
}

thruster_data = pd.read_excel("thruster_data.xlsx", index_col=0)
print(thruster_data)

thruster_name = "NASA GRC 60J"
for key in parameters:
    if not np.isnan(thruster_data.loc[thruster_name][key]):
        parameters[key][2] = thruster_data.loc[thruster_name][key]

optimizer = ThrusterOptimizer(parameters, constants)
optimizer.print_guess_results()

print(parameters["h"][2] / parameters["w"][2] * parameters["E"][2] / parameters["R"][2])
print(parameters["h"][2] / parameters["w"][2] * parameters["E"][2] / (parameters["L"][2] ** (1 / 4) * parameters["R"][2]))

# optimizer.investigate_correlation("h")

# optimizer.print_optimized_results()

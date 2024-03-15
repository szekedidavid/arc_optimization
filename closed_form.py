import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# import pandas as pd


def print_results(input, res):
    print(
        f"E0 = {input[5]} (discharge energy [J])\n"
        f"h = {input[0]} (height [cm])\n"
        f"w = {input[1]} (width [cm])\n"
        f"c = {input[2]} (electrode thickness [cm])\n"
        f"R = {input[3]} (total effective resistance [mOhm])\n"
        f"L = {input[4]} (total effective inductance [nH])\n"
        f"I_EM = {res[0]} (electromagnetic impulse bit [microNs])\n"
        f"I_GD = {res[1]} (gas dynamic impulse bit [microNs])\n"
        f"I_bit = {res[2]} (total impulse bit [microNs])\n"
        f"m_bit = {res[3]} (total mass bit [microg])\n"
        f"fI = {res[4]} (electromagnetic impulse fraction [-])\n"
        f"fm = {res[5]} (electromagnetic mass fraction [-])\n"
        f"I_sp = {res[6]} (specific impulse [s])\n"
        f"eta = {res[7]} (efficiency [-])\n"
        f"R0 = {res[8]} (circuit resistance [mOhm])\n"
        f"Rp = {res[9]} (plasma resistance [mOhm])\n"
        f"E0A = {res[10]} (energy to propellant surface area ratio [J/cm^2])"
    )


class ThrusterOptimizer:
    def __init__(self, params, consts, closed_form=False):
        self.params = params
        self.consts = consts

        self.guess = np.array([self.params[k][2] for k in self.params])
        self.bounds = np.array([(self.params[k][0], self.params[k][1]) for k in self.params])

    def get_results(self, arr: np.ndarray[8], opt: bool):
        I_EM, I_GD, I_bit, m_bit, fI, fm, I_sp, eta, R0, Rp, EpA, I_esp, nu_max, P_max, N_shots, M_prop = self.get_results_open_form(
            arr)

        if opt:
            return 1 / I_esp
        else:
            return I_EM, I_GD, I_bit, m_bit, fI, fm, I_sp, eta, R0, Rp, EpA, I_esp, nu_max, P_max, N_shots, M_prop

    def get_results_open_form(self, arr: np.ndarray[8]):
        h, w, c, R, L, E = arr
        C1, _, C3, g, nu0, um, Rm, n, T_max, I_tot = self.consts.values()
        V = h * 1000
        C = 2 * E / V ** 2

        omega = np.sqrt(1 / (L * 1e-9 * C) - (R * 1e-3) ** 2 / (4 * (L * 1e-9) ** 2))

        def get_RLC_current(t):
            I = V / (omega * L * 1e-9) * np.exp(-R * 1e-3 / (2 * L * 1e-9) * t) * np.sin(omega * t)
            return I

        # x_arr = np.linspace(0, 50e-6, 100)
        # plt.plot(x_arr, get_RLC_current(x_arr))
        # plt.show()

        m_bit = C1 * (h / w) * (E / R) * 1e3
        L_p = nu0 / np.pi * (3 / 2 + np.log((h / w) / (1 + (c / w))))  # inductance gradient
        Int_GD = sp.integrate.quad(lambda t: (get_RLC_current(t) ** 2) ** (1 / n), 0, 50 * 1e-6, limit=200)[0]

        I_EM = E / R * (L_p / 2) * 1e3
        I_GD = C3 * 1e-9 * h / w ** (2 / n - 1) * Int_GD * 1e7
        I_bit = I_EM + I_GD
        I_sp = I_bit * 1e-6 / (m_bit * 1e-9 * g)
        fm = I_EM * 1e-6 / (um * m_bit * 1e-9)  # electromagnetic mass fraction
        fI = I_EM / I_bit  # electromagnetic impulse fraction
        eta = (I_bit * 1e-6) ** 2 / (2 * m_bit * 1e-9 * E)

        # Not sure what to do about plasma inductance
        Rp = (L_p * 10 ** -6 * um / Rm) * 1e3  # plasma resistance
        R0 = R - Rp  # neccessary circuit resistance for effective resistance R

        EpA = E / (h * w)  # energy per propellant surface area, to avoid charring

        I_esp = I_bit / E

        nu_max = T_max / I_bit
        P_max = nu_max * E
        N_shots = I_tot / I_bit * 1e6
        M_prop = N_shots * m_bit * 1e-6

        return I_EM, I_GD, I_bit, m_bit, fI, fm, I_sp, eta, R0, Rp, EpA, I_esp, nu_max, P_max, N_shots, M_prop

    def investigate_correlation(self, var: str):
        """Function that essentially does a sensitivity study.

        Takes the name of an input variable, and varies it between its min and max values from the params dictionary.
        It keeps all other inputs as their respective guess values. Finally, it plots the results."""

        x_arr = np.linspace(self.params[var][0], self.params[var][1], 100)
        results = {key: np.zeros(100) for key in
                   ['I_EM', 'I_GD', 'I_bit', 'm_bit', 'fI', 'fm', 'I_sp', 'eta', 'R0', 'Rp', 'EpA', 'I_esp',
                    'nu_max', 'P_max', 'N_shots', 'M_prop']}
        arr_dict = {key: self.params[key][2] for key in self.params}

        for i in range(np.size(x_arr)):
            arr_dict[var] = x_arr[i]
            arr = np.fromiter(arr_dict.values(), dtype=float)
            results['I_EM'][i], results['I_GD'][i], results['I_bit'][i], results['m_bit'][i], results['fI'][i], \
                results['fm'][i], results['I_sp'][i], results['eta'][i], results['R0'], results['Rp'], \
                results['EpA'][i], results['I_esp'][i], \
                results['nu_max'][i], results['P_max'][i], results['N_shots'][i], results['M_prop'][
                i] = self.get_results(arr, False)

        # fig, axs = plt.subplots(2, 4, figsize=(16, 9))
        # fig.suptitle(f"Varying {var} [cm]")
        #
        # axs[0, 0].plot(x_arr, results['I_EM'], color='b')
        # axs[0, 0].set_title("Electromagnetic impulse bit")
        # axs[0, 0].set_xlabel('$I_{EM} \ [\nu Ns]$')
        #
        # axs[0, 1].plot(x_arr, results['I_GD'], color='b')
        # axs[0, 1].set_title("Gas dynamic impulse bit")
        # axs[0, 1].set_xlabel('$I_{GD} \ [\nu Ns]$')
        #
        # axs[1, 0].plot(x_arr, results['I_bit'], color='b')
        # axs[1, 0].set_title("Total impulse bit")
        # axs[1, 0].set_xlabel('$I_{bit} \ [\nu Ns]$')
        #
        #
        # axs[0, 2].plot(x_arr, results['fI'], color='black')
        # axs[0, 2].set_title("Electromagnetic impulse fraction")
        # axs[0, 2].set_xlabel('$f_I \ [-]$')
        #
        # axs[0, 3].plot(x_arr, results['fm'], color='black')
        # axs[0, 3].set_title("Electromagnetic mass fraction")
        # axs[0, 3].set_xlabel('$f_m \ [-]$')
        #
        # axs[1, 2].plot(x_arr, results['I_sp'], color='black')
        # axs[1, 2].set_title("Specific impulse")
        # axs[1, 2].set_xlabel('$I_{sp} \ [s]$')
        #
        # axs[1, 3].plot(x_arr, results['eta'], color='black')
        # axs[1, 3].set_title("Overall efficiency")
        # axs[1, 3].set_xlabel('$\eta \ [-]$')

        fig, ax = plt.subplots(4, 2)
        ax[0, 0].plot(x_arr, results["I_esp"], color="red")
        ax[0, 0].set_ylabel(r"$I_{sp,P} \ [\mu Ns / J]$")
        ax[0, 1].plot(x_arr, results["I_sp"], color="green")
        ax[0, 1].set_ylabel("$I_{sp,M}$ [s]")
        ax[1, 0].plot(x_arr, results['m_bit'], color='b')
        ax[1, 0].set_ylabel('$m_{bit} \ [\mu g]$')
        ax[1, 1].plot(x_arr, results['I_bit'], color='b')
        ax[1, 1].set_ylabel('$I_{bit} \ [\mu Ns]$')
        ax[2, 0].plot(x_arr, results['N_shots'], color='b')
        ax[2, 0].set_ylabel('$N_{shots} \ [-]$')
        ax[2, 1].plot(x_arr, results['M_prop'], color='b')
        ax[2, 1].set_ylabel('$M_{prop} \ [g]$')
        ax[3, 0].plot(x_arr, results['nu_max'], color='b')
        ax[3, 0].set_ylabel(r'$\nu_{max} \ [Hz]$')
        ax[3, 1].plot(x_arr, results['P_max'], color='b')
        ax[3, 1].set_ylabel('$P_{max} \ [W]$')
        fig.suptitle(f"Influence of varying {var}")

        plt.tight_layout(rect=(0, 0, 1, 1))
        plt.show()

    def optimize_thruster(self):
        """Function running the optimization using SciPy.

        It takes the lower and upper bounds, and the initial guess values, from the params dictionary.
        After finding the solution, it prints both the thruster parameters and the performance values.
        """
        result = sp.optimize.minimize(self.get_results, self.guess, (True,), bounds=self.bounds, method="Nelder-Mead")
        return result

    def print_optimized_results(self):
        """Function that prints the thruster parameters and performance values for the optimized configuration."""

        opt_arr = self.optimize_thruster().x
        opt_res = self.get_results(opt_arr, False)

        print_results(opt_arr, opt_res)

    def print_guess_results(self):
        """Function that prints the thruster parameters and performance values for the initial guess configuration."""

        guess_res = self.get_results(self.guess, False)

        print_results(self.guess, guess_res)


# First and second list elements are the mininum and maxinum values for the optimization,
# and the third ones are the initial guesses (or treated as fixed values, depending on the context).
parameters = {
    "h": [1, 4, 2],  # channel height [cm]
    "w": [0.5, 3, 2],  # electrode width [cm]
    "c": [0.5, 0.5, 0.5],  # electrode thickness [cm]
    "R": [22, 55, 35],  # effective resistance [mOhm]
    "L": [45, 170, 100],  # effetive inductance [nH]
    "E": [5, 60, 10],  # stored energy [J]
}

constants = {
    "C1": 0.0214,
    "C2": 0.134,
    "C4": 0.154,
    "g": 9.80665,
    "nu0": 1.25663706,
    "um": 19.5856e3,  # magnetosonic speed
    "Rm": 2,  # magnetic Reynolds number
    "n": 0.8,  # solid propellant exponent

    "T_max": 2.007,
    "I_tot": 35
}

# thruster_data = pd.read_excel("thruster_data.xlsx", index_col=0)
# print(thruster_data)
#
# thruster_name = "NASA GRC 60J"
# for key in parameters:
#     if not np.isnan(thruster_data.loc[thruster_name][key]):
#         parameters[key][2] = thruster_data.loc[thruster_name][key]

optimizer = ThrusterOptimizer(parameters, constants)
# optimizer.investigate_correlation("E")
optimizer.print_optimized_results()

import numpy as np
import scipy as sp

C1 = 0.0214
C2 = 0.134
g = 9.80665
mu0 = 1.25663706 * 10 ** -6


def get_I_sp(h, w, c, R, L, E0):
    m_bit = C1 * h / w * E0 / R

    Lp = mu0 / np.pi * (3 / 2 + np.log(h / w + c))

    I_bit = E0 / R * (Lp / 2 + C2 / np.sqrt(w) * h / w * (E0 / L) ** (1 / 4))

    I_sp = I_bit / (m_bit * g)

    return 1 / I_sp


result = sp.optimize.minimize(get_I_sp, )

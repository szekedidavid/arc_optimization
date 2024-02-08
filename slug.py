import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


class ThrusterOptimizer:
    def __init__(self, params, consts):
        self.params = params
        self.consts = consts

    def get_results(self, arr: np.ndarray):
        h, w, c, R0, L0, V0, E0 = arr  # todo: what to do about resistance?
        C, g, mu0, um, Rm = self.consts

        C0 = 2 * E0 / V0 ** 2
        Lp = mu0 / np.pi * (3 / 2 + np.log(h / (w + c)))  # todo: is this analytical?
        m_b = C * (h / w) * (E0 / R0)  # todo: use regression instead?

        alpha = R0 * np.sqrt(C0 / L0)
        beta = 2 * np.pi * Lp / mu0
        delta = (V0 * C0) ** 2 / L0 * (mu0 / (2 * np.pi)) ** 2 * 1 / (2 * m_b)


parameters = {
    "h": np.array([1, 4, 4]) * 1e-2,
    "w": np.array([0.5, 3, 1]) * 1e-2,
    "c": np.array([0.3, 1, 0.5]) * 1e-2,
    "R0": np.array([22, 55, 35]) * 1e-3,
    "L0": np.array([45, 170, 100]) * 1e-9,

    "V0": np.array([2000, 2000, 2000]),
    "E0": np.array([20, 20, 20]),
}

constants = {
    "C": 0.0214 * 1e-9,
    "g": 9.80665,
    "mu0": 1.25663706 * 1e-6,
    "um": 20 * 1e3,
    "Rm": 2
}

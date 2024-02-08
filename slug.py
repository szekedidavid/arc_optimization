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
    "h": [1, 4, 4],
    "w": [0.5, 3, 1],
    "c": [0.3, 1, 0.5],
    "R0": [22e-3, 55e-3, 35e-3],
    "L0": [45, 170, 100],

    "V0": [2000, 2000, 2000],
    "E0": [20, 20, 20],
}

constants = {
    "C": 0.0214,
    "g": 9.80665,
    "mu0": 1.25663706,
    "um": 20e3,
    "Rm": 2
}

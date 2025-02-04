from numpy.typing import NDArray
from typing import Tuple
from scipy.optimize import curve_fit
import numpy as np

def reduced_mobility_to_ccs(one_over_k0, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    convert reduced ion mobility (1/k0) to CCS
    :param one_over_k0: reduced ion mobility
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return (SUMMARY_CONSTANT * charge) / (np.sqrt(reduced_mass * (temp + t_diff)) * 1 / one_over_k0)


def ccs_to_one_over_reduced_mobility(ccs, mz, charge, mass_gas=28.013, temp=31.85, t_diff=273.15):
    """
    convert CCS to 1 over reduced ion mobility (1/k0)
    :param ccs: collision cross-section
    :param charge: charge state of the ion
    :param mz: mass-over-charge of the ion
    :param mass_gas: mass of drift gas (N2)
    :param temp: temperature of the drift gas in C°
    :param t_diff: factor to translate from C° to K
    """
    SUMMARY_CONSTANT = 18509.8632163405
    reduced_mass = (mz * charge * mass_gas) / (mz * charge + mass_gas)
    return  ((np.sqrt(reduced_mass * (temp + t_diff))) * ccs) / (SUMMARY_CONSTANT * charge)

def get_sqrt_slopes_and_intercepts(
        mz: NDArray,
        charge: NDArray,
        ccs: NDArray,
        fit_charge_state_one: bool = True,
) -> Tuple[NDArray, NDArray]:
    """
    Fit a sqrt function to the data and return the slopes and intercepts,
    used to parameterize the init layer for the CCS prediction model.
    Args:
        mz: Array of mass-over-charge values
        charge: Array of charge states
        ccs: Array of collision cross-section values
        fit_charge_state_one: Whether to fit the charge state 1 or not (should be set to false if
        your data does not contain charge state 1)

    Returns:
        Tuple of slopes and intercepts the initial projection layer can be parameterized with
    """
    if fit_charge_state_one:
        slopes, intercepts = [], []
    else:
        slopes, intercepts = [0.0], [0.0]

    c_begin = 1 if fit_charge_state_one else 2

    for c in range(c_begin, 6):
        def fit_func(x, a, b):
            return a * np.sqrt(x) + b

        mask = (charge == c)
        mz_tmp = mz[mask]
        ccs_tmp = ccs[mask]

        popt, _ = curve_fit(fit_func, mz_tmp, ccs_tmp)

        slopes.append(popt[0])
        intercepts.append(popt[1])

    return np.array(slopes, np.float32), np.array(intercepts, np.float32)
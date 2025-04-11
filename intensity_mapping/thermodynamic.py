"""
Thermodynamic radiation functions and conversions
steradians are unitless and are added by hand here for recordkeeping

python -m intensity_mapping.thermodynamic
"""
# pylint: disable=no-member, no-name-in-module
import numpy as np
from astropy import constants as const
import astropy.units as u
from intensity_mapping import constants as imcon


def dplanck_dtemp_rj(freq):
    """
    Convert temperature in Rayleigh-Jeans limit to intensity W/m^2/sr/Hz

    >>> print dplanck_dtemp_rj(30. * u.GHz).to(u.Jy / u.sr / u.uK)
    27.6512213648 Jy / (sr uK)

    >>> print dplanck_dtemp_rj(115. * u.GHz).to(u.Jy / u.sr / u.uK)
    406.319336166 Jy / (sr uK)
    """
    conv = 2. * freq ** 2. * const.k_B / const.c ** 2.
    return conv / u.sr


def ysz_to_temp(freq, temperature=imcon.T_CMB):
    """
    Conversion between y_SZ and T_CMB
    g = x coth(x/2) - 4
    Delta T = T_CMB * g

    >>> print ysz_to_temp(150. * u.GHz)
    -2.59823255728 K
    >>> print ysz_to_temp(220. * u.GHz)
    0.104453023754 K
    >>> print ysz_to_temp(353. * u.GHz)
    6.10722185214 K
    """
    planck_x = const.h * freq / (const.k_B * temperature)
    # tanh needs "angular" units, so convert unitless x
    sz_g = planck_x / np.tanh(planck_x * u.rad / 2.) - 4.

    return sz_g * temperature


def ysz_to_sb(freq, temperature=imcon.T_CMB):
    """convert y to surface brightness

    >>> print ysz_to_sb(150. * u.GHz).to(u.MJy / u.sr)
    -1035.337743 MJy / sr
    >>> print ysz_to_sb(220. * u.GHz).to(u.MJy / u.sr)
    50.5068525036 MJy / sr
    >>> print ysz_to_sb(353. * u.GHz).to(u.MJy / u.sr)
    1811.65964437 MJy / sr
    """
    ret = ysz_to_temp(freq, temperature=temperature)
    return ret * dplanck_dtemp(freq, temperature=temperature)


def planck(freq, temperature=None):
    """Planck spectrum W/sr/m^2/Hz

    >>> print planck(90. * u.GHz, temperature=imcon.T_CMB).to(u.MJy / u.sr)
    277.161758343 MJy / sr
    >>> print planck(30. * u.GHz, temperature=imcon.T_CMB).to(u.MJy / u.sr)
    57.2015933295 MJy / sr
    >>> print planck(1000. * u.GHz, temperature=imcon.T_CMB).to(u.MJy / u.sr)
    0.0332077703618 MJy / sr
    """
    planck_exp = np.exp(const.h * freq / (const.k_B * temperature))
    ret = 1. / (planck_exp - 1.)
    ret *= 2. * const.h * freq ** 3. / const.c ** 2.

    return ret / u.sr


def dplanck_dtemp(freq, temperature=imcon.T_CMB):
    """
    The first order conversion factor between CMB temperature units (K_CMB) and
    spectral intensity units (W/m^2 sr Hz), dB/dT.

    Since this is a linear conversion (in intensity or temperature units), one
    may get the inverse conversion with 1/dBdT.

    Returns the spectral intensity per CMB temperature in (W/m^2 sr Hz)/K_CMB.

    >>> print dplanck_dtemp(200 * u.GHz).to(u.Jy / u.uK / u.sr)
    478.213344642 Jy / (sr uK)

    >>> print dplanck_dtemp(30 * u.GHz).to(u.Jy / u.uK / u.sr)
    27.0170595751 Jy / (sr uK)

    >>> print dplanck_dtemp(115 * u.GHz).to(u.Jy / u.uK / u.sr)
    291.893182434 Jy / (sr uK)

    # A simple calculation to compare to
    # https://arxiv.org/pdf/1609.08942.pdf and planck XXX

    # the actual bandpass is 287.45 MJy / K
    >>> print dplanck_dtemp(353 * u.GHz).to(u.MJy / u.K / u.sr)
    296.642186616 MJy / (K sr)

    # the actual bandpass is 58.04 MJy / K
    >>> print dplanck_dtemp(545 * u.GHz).to(u.MJy / u.K / u.sr)
    57.1137565535 MJy / (K sr)

    # the actual bandpass is 2.27 MJy / K
    >>> print dplanck_dtemp(857 * u.GHz).to(u.MJy / u.K / u.sr)
    1.43558498529 MJy / (K sr)
    """
    planck_exp = np.exp(const.h * freq / (const.k_B * temperature))
    ret = 2 * const.h ** 2. * (freq ** 4.)
    ret /= const.k_B * const.c ** 2. * temperature * temperature
    ret *= planck_exp / ((planck_exp - 1.) ** 2.)

    return ret / u.sr


if __name__ == "__main__":
    import doctest

    doctest.testmod()

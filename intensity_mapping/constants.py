import astropy.units as u
from astropy.cosmology import Planck15 as cosmo

#speed_c = 299792458.      # m/s
#planck_h = 6.62606957e-34  # m^2*kg/s
#kB = 1.3806488e-23  # m^2*kg/s^2 K
#MJypsr = 1.e-20     # (W/m^2 sr Hz) per MJy/sr
#kJypsr = 1.e-23     # (W/m^2 sr Hz) per kJy/sr
#Jypsr = 1.e-26     # (W/m^2 sr Hz) per Jy/sr

T_CMB = 2.72548 * u.K

# TODO: unify these tables
# Frequency of the lines of interest in GHz
def co_freq(j_quantum_end):
    """simple form for the CO J lines"""
    return 115. * (j_quantum_end + 1)

nu0_lines = {"CII": 1897. * u.GHz}

#correct CII line frequency.
nu0_lines["CII"] = .299792/(157.7*10**-6)

for j_end in range(15):
    line_label = "CO%d-%d" % (j_end + 1, j_end)
    nu0_lines[line_label] = 115. * (j_end + 1) * u.GHz


#nu0_lines["HI"] = 1420. * u.MHz

root_data = "data/"

#pixie_fwhm = 2.6 * u.deg
pixie_fwhm = 1.65 * u.deg

f_sky = {}
f_sky["CMASS_North"] = 0.18309
f_sky["CMASS_South"] = 0.069499
f_sky["LOWZ_North"] = 0.15996
f_sky["LOWZ_South"] = 0.069504

boss_bias = 2.
cii_bias = 2.
cii_brightness = 2.

# also given Ode0
hubble = cosmo.H0.value / 100.
#cosmo_params = {"h": hubble,
#                "ombh2": cosmo.Ob0 * hubble ** 2.,
#                "omch2": cosmo.Odm0 * hubble ** 2.,
#                "num_massive_neutrinos": 0,
#                "mnu":0.,
#                "Neff":cosmo.Neff,
#                "omk":0.,
#                "ns":0.9667,
#                "As":2.441e-9}

#Patrick has As=2.142e-9 (Planck 2015)
#And ns=0.9645

#Planck 2015 TT + lowP

import numpy as np

cosmo_params = {"h": 0.6731,
                "ombh2": 0.02222,
                "omch2": 0.1197,
                "num_massive_neutrinos": 0,
                "mnu":0.,
                "Neff":cosmo.Neff,
                "omk":0.,
                "ns":0.9655,
                "As":np.exp(3.089)/(10**10)}

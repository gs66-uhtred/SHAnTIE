"""
Calculate matter power spectra with CAMB and CLASS

CAMB and CLASS have different parameter names. These provide convenience
functions that wrap the astropy cosmological parameters.

TODO:
compare with LAMBDA CAMB output
CLASS curvature convention
"""
# pylint: disable=no-member, no-name-in-module
import numpy as np
import astropy.units as u
from . import constants as con


def camb_pk(kvec, params, redshift=0., nonlinear=True):
    """Given k in h Mpc^-1, return Pk in h^-3 Mpc^3
    using the CAMB anisotropy code

    >>> kvec = np.array([0.01, 0.1, 1.]) / u.Mpc
    >>> print camb_pk(kvec, con.cosmo_params, redshift=0., nonlinear=True)
    [ 26094.1...   6577.44...    513.746...] Mpc3
    >>> print camb_pk(kvec, con.cosmo_params, redshift=0., nonlinear=False)
    [ 26291.3...   6448.31...     79.6165...] Mpc3
    >>> print camb_pk(kvec, con.cosmo_params, redshift=1., nonlinear=True)
    [ 9731.63...  2419.31...    105.936...] Mpc3
    >>> print camb_pk(kvec, con.cosmo_params, redshift=1., nonlinear=False)
    [ 9760.77...  2393.97...    29.5580...] Mpc3
    """
    import camb
    from camb import model
    # see CAMB's pycamb/camb_tests/camb_test.py
    pars = camb.CAMBparams()

    pars.set_cosmology(H0=params['h'] * 100,
                       ombh2=params['ombh2'],
                       omch2=params['omch2'],
                       num_massive_neutrinos=params['num_massive_neutrinos'],
                       mnu=params['mnu'],
                       nnu=params['Neff'],
                       omk=params['omk'])

    pars.set_dark_energy()  # re-set defaults
    pars.InitPower.set_params(ns=params['ns'], As=params['As'])

    pk_func = camb.get_matter_power_interpolator(pars, nonlinear=nonlinear)
    kmax = 1.1 * np.max(kvec).to(1. / u.Mpc).value
    pars.set_matter_power(redshifts=[redshift],
                          kmax=kmax, k_per_logint=5)

    if nonlinear:
        pars.NonLinear = model.NonLinear_pk

    return pk_func.P(redshift, kvec.to(1. / u.Mpc).value) * u.Mpc ** 3.


def class_pk(kvec, params, redshift=0., nonlinear=True):
    """Given k in h Mpc^-1, return Pk in h^-3 Mpc^3
    using the CLASS anisotropy code
    TODO: implement massive neutrinos, nonlinear?

    >>> kvec = np.array([0.01, 0.1, 1.]) / u.Mpc
    >>> print class_pk(kvec, con.cosmo_params, redshift=0., nonlinear=True)
    [ 26090.0...   6586.96...    513.450...] Mpc3
    >>> print class_pk(kvec, con.cosmo_params, redshift=0., nonlinear=False)
    [ 26287.5...   6458.68...     79.6092...] Mpc3
    >>> print class_pk(kvec, con.cosmo_params, redshift=1., nonlinear=True)
    [ 9730.07...  2422.99...    106.09...] Mpc3
    >>> print class_pk(kvec, con.cosmo_params, redshift=1., nonlinear=False)
    [ 9759.29...  2397.77...    29.5548...] Mpc3
    """
    from classy import Class
    params = {
        'output': 'mPk',
        'lensing': 'no',
        'P_k_max_h/Mpc': 10.,
        'A_s': params['As'],
        'n_s': params['ns'],
        'h': params['h'],
        'z_pk': redshift,
        'N_ur': params['Neff'],
        'Omega_k': params['omk'],
        'omega_b': params['ombh2'],
        'omega_cdm': params['omch2']}

    if nonlinear:
        params["non linear"] = "halofit"

    hubble = params['h']
    clpk = Class()
    clpk.set(params)
    clpk.compute()

    kvec_eval = kvec.to(1. / u.Mpc).value * hubble
    power_k = np.array([clpk.pk(kval, redshift) for kval in kvec_eval])

    return power_k * hubble ** 3. * u.Mpc ** 3.


def _load_pk(pkfile, k_vec):
    """Helper function
    Useful for power spectra from CAMB tool at LAMBDA
    power spectrum with k in h Mpc^-1 and power in h^-3 Mpc^3
    """
    pkz1_data = np.loadtxt(pkfile)
    pkz1_k = pkz1_data[:, 0] / u.Mpc
    pkz1 = pkz1_data[:, 1] * u.Mpc ** 3.
    return np.interp(k_vec.to(1. / u.Mpc).value, pkz1_k, pkz1) * u.Mpc ** 3.


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)

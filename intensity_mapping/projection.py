"""
Functions for projecting 3D power onto 2D

TODO:
will Limber functions work in curved universe?
pass in cosmological parameters
"""
# pylint: disable=no-member, no-name-in-module
import astropy.units as u
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from scipy import interpolate, integrate, special
import scipy as sp
import astropy
from classy import Class
from . import constants as con
from . import power_spectra as ps


def growth_function(redshifts):
    """
    Calculate the cosmological linear growth function
    From: randomfield/cosmotools.html

    Linder 2005, Eq. 14-16 Weinberg 2012
    exp(-\int Omega_m^0.55 / (1+z) dz)

    >>> print growth_function(np.array([0., 0.1, 1., 5.]))
    [ 1.          0.9488565   0.61529236  0.18511194]
    """
    z_axis = np.arange(3)[np.argsort(redshifts.shape)][-1]
    gamma = 0.55
    integrand = np.power(cosmo.Om(redshifts), gamma) / (1 + redshifts)
    exponent = sp.integrate.cumtrapz(y=integrand, x=redshifts,
                                     axis=z_axis, initial=0)

    return np.exp(-exponent)


def _config_classgal(nonlinear=True, limber=False):
    """Helper to return CLASS configuration for projected power spectra
    """
    params = {
        'output': 'dCl',
        'lensing': 'no',
        "selection_sampling_bessel": 2.,
        "selection_tophat_edge": 0.001,
        "k_step_trans_scalars": 0.4,
        "k_scalar_max_tau0_over_l_max": 2.,
        'A_s': con.cosmo_params['As'],
        'n_s': con.cosmo_params['ns'],
        'h': con.cosmo_params['h'],
        'N_ur': con.cosmo_params['Neff'],
        'Omega_k': con.cosmo_params['omk'],
        'omega_b': con.cosmo_params['ombh2'],
        'omega_cdm': con.cosmo_params['omch2']}

    if limber:
        params["l_switch_limber_for_nc_local_over_z"] = 0.
        params["l_switch_limber_for_nc_los_over_z"] = 0.
    else:
        params["l_switch_limber_for_nc_local_over_z"] = 10000.
        params["l_switch_limber_for_nc_los_over_z"] = 10000.

    if nonlinear:
        params["non linear"] = "halofit"

    return params


def _config_classgal_pk(z_pk = 0, nonlinear=True, limber=False):
    """Helper to return CLASS configuration for projected power spectra
    """
    params = {
        'output': 'mPk, mTk',
        'lensing': 'no',
        "selection_sampling_bessel": 2.,
        "selection_tophat_edge": 0.001,
        "k_step_trans_scalars": 0.4,
        "k_scalar_max_tau0_over_l_max": 2.,
        'A_s': con.cosmo_params['As'],
        'n_s': con.cosmo_params['ns'],
        'h': con.cosmo_params['h'],
        'N_ur': con.cosmo_params['Neff'],
        'Omega_k': con.cosmo_params['omk'],
        'omega_b': con.cosmo_params['ombh2'],
        'omega_cdm': con.cosmo_params['omch2']}
    params['z_pk'] = z_pk

    if limber:
        params["l_switch_limber_for_nc_local_over_z"] = 0.
        params["l_switch_limber_for_nc_los_over_z"] = 0.
    else:
        params["l_switch_limber_for_nc_local_over_z"] = 10000.
        params["l_switch_limber_for_nc_los_over_z"] = 10000.

    if nonlinear:
        params["non linear"] = "halofit"

    return params

# -----------------------------------------------------------------------------
# 3D to 2D projection for Gaussian kernels
# -----------------------------------------------------------------------------

def classgal_gauss(ell_range, mean, sigma, nonlinear=True, limber=False):
    """Exact Cl integration facility from CLASSGal

    Note that the limber options are specific to newer versions of class
    Upgrade if l_switch_limber is failing

    >>> ell_range = np.arange(100, 104)
    >>> cl = classgal_gauss(ell_range, 1., 0.1, nonlinear=False, limber=False)
    >>> print cl
    [  1.93341...e-06   1.91589...e-06   1.89885...e-06   1.88228...e-06]

    >>> ell_range = np.arange(100, 104)
    >>> cl = classgal_gauss(ell_range, 1., 0.1, nonlinear=True, limber=False)
    >>> print cl
    [  1.91716...e-06   1.89982...e-06   1.88296...e-06   1.86656...e-06]

    >>> ell_range = np.arange(100, 104)
    >>> cl = classgal_gauss(ell_range, 1., 0.1, nonlinear=False, limber=True)
    >>> print cl
    [  1.94136...-06   1.92353...e-06   1.90619...e-06   1.88933...e-06]

    >>> ell_range = np.arange(100, 104)
    >>> cl = classgal_gauss(ell_range, 1., 0.1, nonlinear=True, limber=True)
    >>> print cl
    [  1.92503...e-06   1.90738...e-06   1.89022...e-06   1.87354...e-06]
    """
    lmax = np.max(ell_range) + 1

    params = _config_classgal(nonlinear=nonlinear, limber=limber)
    params["selection"] = "gaussian"
    params["selection_mean"] = mean
    params["selection_width"] = sigma
    params["l_max_lss"] = lmax

    clpk = Class()
    clpk.set(params)
    clpk.compute()

    cl_out = clpk.density_cl(lmax)
    cl = cl_out['dd'][0]
    ell = cl_out['ell']

    return cl[ell_range]


def limber_gauss(ell_range, z_mean, z_sigma, nonlinear=True):
    """Limber integral between with given mean and sigma

    >>> ell_range = np.arange(100, 104)
    >>> print limber_gauss(ell_range, 1., 0.1, nonlinear=False)
    [  1.94676...e-06   1.92858...e-06   1.91090...e-06   1.89371...e-06]
    >>> print limber_gauss(ell_range, 1., 0.1, nonlinear=True)
    [  1.93064...e-06   1.91265...e-06   1.89515...e-06   1.87814...e-06]
    """
    kvec = 10. ** np.linspace(-4., 1., 5000.) / u.Mpc
    pk_bins = ps.camb_pk(kvec, con.cosmo_params, redshift=z_mean,
                         nonlinear=nonlinear)

    logk = np.log10(kvec.value)
    logp = np.log10(pk_bins.value)
    pinterp = interpolate.interp1d(logk, logp)

    def norm(x, mean, sigma):
        fac = np.exp(-(x - mean) ** 2. / 2. / sigma ** 2.)
        return fac / np.sqrt(2. * sigma ** 2. * np.pi)

    def limber_integrand(z_int):
        hubble = cosmo.H(z_int) / cosmo.h / astropy.constants.c.to('km/s')
        chi = cosmo.comoving_distance(z_int) * cosmo.h

        geom = hubble / chi ** 2.
        klimber = float(ell_eval) / chi
        power = np.power(10., pinterp(np.log10(klimber.value))) * u.Mpc ** 3.
        return geom * power * norm(z_int, z_mean, z_sigma) ** 2.

    cls = np.zeros_like(ell_range, dtype=float)

    z_start = z_mean - 5. * z_sigma
    z_end = z_mean + 5. * z_sigma

    for index, ell_eval in enumerate(ell_range):
        if ell_eval == 0 or ell_eval == 1:
            cls[index] = 0.
        else:
            cls[index] = sp.integrate.quad(limber_integrand,
                                           z_start,
                                           z_end)[0]

    return cls


# -----------------------------------------------------------------------------
# 3D to 2D projection for top-hat kernels
# -----------------------------------------------------------------------------

def classgal_tophat(ell_range, z_start, z_end, nonlinear=True, limber=False):
    """Exact Cl integration facility from CLASSGal

    Note that the limber options are specific to newer versions of class
    Upgrade if l_switch_limber is failing

    >>> ell_range = np.arange(100, 104)
    >>> cl = classgal_tophat(ell_range, 1., 1.1, nonlinear=False, limber=False)
    >>> print cl
    [  5.9177...e-06   5.8684...e-06   5.8202...e-06   5.7733...e-06]

    >>> ell_range = np.arange(100, 104)
    >>> cl = classgal_tophat(ell_range, 1., 1.1, nonlinear=True, limber=False)
    >>> print cl
    [  5.8746...e-06   5.8258...e-06   5.7781...e-06   5.7316...e-06]

    >>> ell_range = np.arange(100, 104)
    >>> cl = classgal_tophat(ell_range, 1., 1.1, nonlinear=False, limber=True)
    >>> print cl
    [  6.7698...e-06   6.6807...e-06   6.5797...e-06   6.4644...e-06]

    >>> ell_range = np.arange(100, 104)
    >>> cl = classgal_tophat(ell_range, 1., 1.1, nonlinear=True, limber=True)
    >>> print cl
    [  6.7163...e-06   6.6280...e-06   6.5278...e-06   6.4135...e-06]
    """
    if (z_end < 0.) or (z_start < 0.):
        return None

    lmax = np.max(ell_range) + 1

    params = _config_classgal(nonlinear=nonlinear, limber=limber)
    params["selection"] = "tophat"
    params["selection_mean"] = (z_start + z_end) / 2.
    params["selection_width"] = np.abs((z_start - z_end) / 2.)
    params["l_max_lss"] = lmax

    clpk = Class()
    clpk.set(params)
    clpk.compute()

    cl_out = clpk.density_cl(lmax)
    cl = cl_out['dd'][0]
    ell = cl_out['ell']

    return cl[ell_range]

def classgal_tophat_zpairs(ell_range, z_start, z_end, nonlinear=True, limber=False, ignore_rsd_lensing=False, bias=1, make_bias_string=False, num_non_diag = None):
    """Exact Cl integration facility from CLASSGal

    Note that the limber options are specific to newer versions of class
    Upgrade if l_switch_limber is failing
    """
    z_start = np.array(z_start)
    z_end = np.array(z_end)
    z_centers = (z_start + z_end) / 2.
    z_widths = np.abs((z_start - z_end) / 2.)
    z_centers_str = str(z_centers[0])
    z_widths_str = str(z_widths[0])
    for cent, width in zip(z_centers[1:],z_widths[1:]):
        z_centers_str += ',' + str(cent)
        z_widths_str += ',' + str(width)

    print(z_centers)
    lmax = np.max(ell_range) + 1

    params = _config_classgal(nonlinear=nonlinear, limber=limber)
    params["selection"] = "tophat"
    params["selection_mean"] = z_centers_str
    params["selection_width"] = z_widths_str
    if type(num_non_diag) == type(None):
        params["non_diagonal"] = len(z_centers) - 1
    else:
        params["non_diagonal"] = num_non_diag
    params["l_max_lss"] = lmax
    #params["lensing"] = "no"
    if ignore_rsd_lensing:
        params["number count contributions"] = "density"
    else:
        params["number count contributions"] = "density, rsd, lensing"
    if not make_bias_string:
        params["selection_bias"] = bias
    else:
        bias_str = str(bias[0])
        for b in bias[1:]:
            bias_str += ',' + str(b)
        params["selection_bias"] = bias_str

    print(params)
    clpk = Class()
    clpk.set(params)
    clpk.compute()

    cl_out = clpk.density_cl(lmax)
    return cl_out


def fill_cl_matrix_from_dict(cl_dict, cut_last_ell=True, num_non_diag = None):
    #For full off-diagonal info
    if type(num_non_diag) == type(None):
        dict_len = len(list(cl_dict.keys()))
        z_len = int((-1 + (1+8*dict_len)**0.5)/2)
        ell_len = len(cl_dict[0])
        if cut_last_ell:
            ell_len -= 1
        indices = np.triu_indices(z_len)
        cl_mat = np.zeros((ell_len, z_len, z_len))
        for ind in range(dict_len):
            cl_mat[:,indices[0][ind],indices[1][ind]] = cl_dict[ind][:ell_len]
            cl_mat[:,indices[1][ind], indices[0][ind]] = cl_dict[ind][:ell_len]
    else:
        #Only block-diagonal cl computed.
        dict_len = len(list(cl_dict.keys()))
        z_len = int((dict_len + (num_non_diag**2 + num_non_diag)/2)/(1+num_non_diag))
        ell_len = len(cl_dict[0])
        if cut_last_ell:
            ell_len -= 1
        cl_mat = np.zeros((ell_len, z_len, z_len))
        for ind in range(z_len - num_non_diag):
            for off_diag in range(num_non_diag + 1):
                cl_mat[:,ind,ind+off_diag] = cl_dict[ind*(num_non_diag+1)+off_diag][:ell_len]
                cl_mat[:,ind+off_diag,ind] = cl_dict[ind*(num_non_diag+1)+off_diag][:ell_len]
        offset = 0
        starting_index = z_len - num_non_diag
        for ind in range(z_len - num_non_diag, z_len): 
            for off_diag in range(z_len - ind):
                cl_mat[:,ind,ind+off_diag] = cl_dict[offset + starting_index*(num_non_diag+1)][:ell_len]
                cl_mat[:,ind+off_diag,ind] = cl_dict[offset + starting_index*(num_non_diag+1)][:ell_len]
                offset += 1
    return cl_mat

def limber_tophat(ell_range, z_start, z_end, nonlinear=True):
    """Limber integral between in a tophat between z_start and z_end

    >>> ell_range = np.arange(100, 104)
    >>> print limber_tophat(ell_range, 1., 1.1, nonlinear=False)
    [  6.83024...e-06   6.76430...e-06   6.69991...e-06   6.63707...e-06]
    >>> print limber_tophat(ell_range, 1., 1.1, nonlinear=True)
    [  6.77315...e-06   6.70786...e-06   6.64412...e-06   6.58193...e-06]
    """
    if (z_end < 0.) or (z_start < 0.):
        return None

    z_ref = (z_start + z_end) / 2.
    delta_z = z_end - z_start

    kvec = 10. ** np.linspace(-4., 1., 5000.) / u.Mpc
    pk_bins = ps.camb_pk(kvec, con.cosmo_params, redshift=z_start,
                         nonlinear=nonlinear)

    logk = np.log10(kvec.value)
    logp = np.log10(pk_bins.value)
    pinterp = interpolate.interp1d(logk, logp)

    def limber_integrand(z_int):
        hubble = cosmo.H(z_int) / cosmo.h / astropy.constants.c.to('km/s')
        chi = cosmo.comoving_distance(z_int) * cosmo.h

        geom = hubble / chi ** 2.
        klimber = float(ell_eval) / chi
        power = np.power(10., pinterp(np.log10(klimber.value))) * u.Mpc ** 3.
        return geom * power

    cls = np.zeros_like(ell_range, dtype=float)

    for index, ell_eval in enumerate(ell_range):
        if ell_eval == 0 or ell_eval == 1:
            cls[index] = 0.
        else:
            cls[index] = sp.integrate.quad(limber_integrand,
                                           z_start,
                                           z_end)[0]

    return cls / delta_z ** 2.


# -----------------------------------------------------------------------------
# 3D to 2D projection for two arbitrary kernels
# -----------------------------------------------------------------------------

def limber_generic(ell_range, sel1, sel2, bias_multiplier, nonlinear=True,
                   z_start=0., z_end=6., skip_norm1=False, skip_norm2=False):
    """Permits aribtrary selection functions and b(k) multiplier to be
    handed in to the Limber integral.

    Uses a simple growth function to move between redshifts, so
    nonlinear=True is violated; it is evaluated at z_start

    It is also possible to evaluate the power spectrum at z=0 and scale by
    growth to all previous redshifts, but this results in much lower accuracy
    in the nonlinear part of the spectrum.

    Warning: for sharp selection functions, the z range needs to be close to
    the band edges or else the integrator will not see the region.
    """
    kvec = 10. ** np.linspace(-4., 1., 5000.) / u.Mpc

    z_eval = (z_start + z_end) / 2.
    pk_bins = ps.camb_pk(kvec, con.cosmo_params, redshift=z_eval,
                         nonlinear=nonlinear)

    logk = np.log10(kvec.value)
    logp = np.log10(pk_bins.value)
    pinterp = interpolate.interp1d(logk, logp)

    norm1 = sp.integrate.quad(sel1, z_start, z_end)[0]
    norm2 = sp.integrate.quad(sel2, z_start, z_end)[0]

    if skip_norm1:
        norm1 = 1.

    if skip_norm2:
        norm2 = 1.

    z_range = np.linspace(0, z_end, 1000)
    growth = growth_function(z_range)
    growth_f = interpolate.interp1d(z_range, growth)

    def limber_integrand(z_int):
        hubble = cosmo.H(z_int) / cosmo.h / astropy.constants.c.to('km/s')
        chi = cosmo.comoving_distance(z_int) * cosmo.h

        geom = hubble / chi ** 2.
        klimber = float(ell_eval) / chi
        # This only works for scalar z_int, as used in quad()
        try:
            logp_klimber = pinterp(np.log10(klimber.value))
            power = np.power(10., logp_klimber) * u.Mpc ** 3.
        except ValueError:
            power = 0. * u.Mpc ** 3.

        power *= bias_multiplier(klimber)

        # assuming that the power spectrum already has a growth multiplier
        # that was evalued at the central redshift
        power = (growth_f(z_int) / growth_f(z_eval)) ** 2. * power

        selection = sel1(z_int) * sel2(z_int) / (norm1 * norm2)

        return geom * power * selection

    cls = np.zeros_like(ell_range, dtype=float)

    for index, ell_eval in enumerate(ell_range):
        if ell_eval == 0 or ell_eval == 1:
            cls[index] = 0.
        else:
            cls[index] = sp.integrate.quad(limber_integrand,
                                           z_start,
                                           z_end)[0]

    return cls


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)

"""
Provide a common interface to large scale structure catalogs.
Doctests here load each catalog and may be slow.

TODO:
Get DR12 QSOs (currently pre-release directory)
Do the healpix shift in rebinning; same answers?
Convert from rad to Lat/Lon and print hms dms; confirm
print Field center in HMS, DMS
"""
# pylint: disable=no-member, no-name-in-module
import h5py
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import Longitude
from astropy.coordinates import Latitude

# -----------------------------------------------------------------------------
# Paths to LSS galaxy sample
# -----------------------------------------------------------------------------
DATA_ROOT = '/Volumes/data_drive/data/'
#DATA_ROOT = '/home/eswitzer/data/'

BOSS7_DIR = DATA_ROOT + 'BOSS/dr7/'
BOSS10_DIR = DATA_ROOT + 'BOSS/dr10/boss/'
BOSS12_DIR = DATA_ROOT + 'BOSS/dr12/boss/'

TWOMPZ_DIR = DATA_ROOT + '2MASS_combined_photoz/'

# -----------------------------------------------------------------------------
# BOSS DR10 files
# use load_boss10()
# -----------------------------------------------------------------------------
BOSS10_CMASS_NORTH = BOSS10_DIR + 'lss/galaxy_DR10v8_CMASS_North.fits.gz'
BOSS10_CMASS_NORTH_RAND = BOSS10_DIR + 'lss/random2_DR10v8_CMASS_North.fits.gz'
BOSS10_CMASS_SOUTH = BOSS10_DIR + 'lss/galaxy_DR10v8_CMASS_South.fits.gz'
BOSS10_CMASS_SOUTH_RAND = BOSS10_DIR + 'lss/random2_DR10v8_CMASS_South.fits.gz'
BOSS10_LOWZ_NORTH = BOSS10_DIR + 'lss/galaxy_DR10v8_LOWZ_North.fits.gz'
BOSS10_LOWZ_NORTH_RAND = BOSS10_DIR + 'lss/random2_DR10v8_LOWZ_North.fits.gz'
BOSS10_LOWZ_SOUTH = BOSS10_DIR + 'lss/galaxy_DR10v8_LOWZ_South.fits.gz'
BOSS10_LOWZ_SOUTH_RAND = BOSS10_DIR + 'lss/random2_DR10v8_LOWZ_South.fits.gz'


def load_boss10(filename):
    """Load BOSS DR10 data
    >>> ra, dec, z = load_boss10(BOSS10_CMASS_NORTH)
    >>> print ra[0], dec[0], z[0]
    2.25454980359 rad 0.716518757642 rad 0.542531
    >>> print ra[-1], dec[-1], z[-1]
    4.38886503518 rad 0.835782087979 rad 0.452405

    >>> ra, dec, z = load_boss10(BOSS10_CMASS_SOUTH)
    >>> print ra[0], dec[0], z[0]
    5.61089021423 rad 1.39672696363 rad 0.301395
    >>> print ra[-1], dec[-1], z[-1]
    5.7352246679 rad 1.54888589087 rad 0.541101

    >>> ra, dec, z = load_boss10(BOSS10_LOWZ_NORTH)
    >>> print ra[0], dec[0], z[0]
    3.40708707185 rad 1.57961474884 rad 0.345882
    >>> print ra[-1], dec[-1], z[-1]
    4.46391538717 rad 0.870056029801 rad 0.424335

    >>> ra, dec, z = load_boss10(BOSS10_LOWZ_SOUTH)
    >>> print ra[0], dec[0], z[0]
    5.60801097492 rad 1.39747682475 rad 0.411104
    >>> print ra[-1], dec[-1], z[-1]
    5.75673600867 rad 1.54895804462 rad 0.108355
    """
    gal_data = fits.open(filename)

    #gal_dec = -gal_data[1].data['PLUG_DEC'][:] * np.pi / 180. + np.pi / 2.
    #gal_ra = gal_data[1].data['PLUG_RA'][:] * np.pi / 180.
    gal_dec = -gal_data[1].data['DEC'][:] * np.pi / 180. + np.pi / 2.
    gal_ra = gal_data[1].data['RA'][:] * np.pi / 180.
    gal_z = gal_data[1].data['Z'][:]

    gal_data.close()

    return gal_ra * u.rad, gal_dec * u.rad, gal_z

# -----------------------------------------------------------------------------
# BOSS DR12 files
# use load_boss12()
# -----------------------------------------------------------------------------
BOSS12_CMASS_NORTH = BOSS12_DIR + 'lss/galaxy_DR12v5_CMASS_North.fits.gz'
BOSS12_CMASS_NORTH_RAND = BOSS12_DIR + 'lss/random2_DR12v5_CMASS_North.fits.gz'
BOSS12_CMASS_SOUTH = BOSS12_DIR + 'lss/galaxy_DR12v5_CMASS_South.fits.gz'
BOSS12_CMASS_SOUTH_RAND = BOSS12_DIR + 'lss/random2_DR12v5_CMASS_South.fits.gz'
BOSS12_LOWZ_NORTH = BOSS12_DIR + 'lss/galaxy_DR12v5_LOWZ_North.fits.gz'
BOSS12_LOWZ_NORTH_RAND = BOSS12_DIR + 'lss/random2_DR12v5_LOWZ_North.fits.gz'
BOSS12_LOWZ_SOUTH = BOSS12_DIR + 'lss/galaxy_DR12v5_LOWZ_South.fits.gz'
BOSS12_LOWZ_SOUTH_RAND = BOSS12_DIR + 'lss/random2_DR12v5_LOWZ_South.fits.gz'


def load_boss12(filename):
    """Load BOSS DR12 data

    >>> ra, dec, z = load_boss12(BOSS12_CMASS_NORTH)
    >>> print ra[0], dec[0], z[0]
    2.25454980359 rad 0.716518757642 rad 0.54253
    >>> print ra[-1], dec[-1], z[-1]
    3.75536872984 rad 0.435609016141 rad 0.485066

    >>> ra, dec, z = load_boss12(BOSS12_CMASS_SOUTH)
    >>> print ra[0], dec[0], z[0]
    5.61089021423 rad 1.39672696363 rad 0.301401
    >>> print ra[-1], dec[-1], z[-1]
    5.7352246679 rad 1.54888589087 rad 0.541101

    >>> ra, dec, z = load_boss12(BOSS12_LOWZ_NORTH)
    >>> print ra[0], dec[0], z[0]
    3.40708707185 rad 1.57961474884 rad 0.345888
    >>> print ra[-1], dec[-1], z[-1]
    3.92151744093 rad 0.474067627848 rad 0.208536

    >>> ra, dec, z = load_boss12(BOSS12_LOWZ_SOUTH)
    >>> print ra[0], dec[0], z[0]
    5.60801097492 rad 1.39747682475 rad 0.411108
    >>> print ra[-1], dec[-1], z[-1]
    5.75673600867 rad 1.54895804462 rad 0.108355
    """
    gal_data = fits.open(filename)

    gal_dec = -gal_data[1].data['DEC'][:] * np.pi / 180. + np.pi / 2.
    gal_ra = gal_data[1].data['RA'][:] * np.pi / 180.
    gal_z = gal_data[1].data['Z'][:]

    gal_data.close()

    return gal_ra * u.rad, gal_dec * u.rad, gal_z


# -----------------------------------------------------------------------------
# BOSS QSO files (DR7, DR10, DR12)
# use load_qso() for DR10, 12 and load_qso7 for legacy DR7
# http://data.sdss3.org/sas/dr12/boss/qso/DR12Q/
# http://classic.sdss.org/dr7/products/value_added/qsocat_dr7.html
# -----------------------------------------------------------------------------
BOSS10_QSO = BOSS10_DIR + 'qso/DR10Q/DR10Q_v2.fits'
#BOSS12_QSO = BOSS12_DIR + 'qso/DR12Q/DR12Q.fits'
BOSS12_QSO = DATA_ROOT + "BOSS/dr12_prerel/boss/" + 'qso/DR12Q/DR12Q.fits'
BOSS7_QSO = BOSS7_DIR + 'dr7qso.fit.gz'


def load_qso(filename):
    """Load BOSS DR10, DR12 QSO data

    >>> ra, dec, z = load_qso(BOSS10_QSO)
    >>> print ra[0], dec[0], z[0]
    7.06812586...e-05 rad 1.48650072... rad 1.62829276...
    >>> print ra[-1], dec[-1], z[-1]
    6.28312024... rad 1.52160333... rad 2.37787762...

    >>> ra, dec, z = load_qso(BOSS12_QSO)
    >>> print ra[0], dec[0], z[0]
    3.31313265...e-05 rad 1.26058605... rad 2.30763868...
    >>> print ra[-1], dec[-1], z[-1]
    6.28318429... rad 0.96434455... rad 2.363834...
    """
    gal_data = fits.open(filename)

    gal_dec = -gal_data[1].data['DEC'][:] * np.pi / 180. + np.pi / 2.
    gal_ra = gal_data[1].data['RA'][:] * np.pi / 180.
    gal_z = gal_data[1].data['Z_PCA'][:]

    return gal_ra * u.rad, gal_dec * u.rad, gal_z


def load_qso7(filename):
    """Load BOSS DR7 QSO data

    >>> ra, dec, z = load_qso7(BOSS7_QSO)
    >>> print ra[0], dec[0], z[0]
    0.000475218248... rad 1.56180192... rad 1.8246
    >>> print ra[-1], dec[-1], z[-1]
    6.28314472... rad 1.56826786... rad 1.3542
    """
    gal_data = fits.open(filename)

    gal_dec = -gal_data[1].data['DEC'][:] * np.pi / 180. + np.pi / 2.
    gal_ra = gal_data[1].data['RA'][:] * np.pi / 180.
    gal_z = gal_data[1].data['Z'][:]

    return gal_ra * u.rad, gal_dec * u.rad, gal_z


# -----------------------------------------------------------------------------
# TwoMPZ files
# use load_twompz()
# Pulled from db http://surveys.roe.ac.uk/ssa/sql.html
# SELECT * FROM TWOMPZ..twompzPhotoz
# -----------------------------------------------------------------------------
TWOMPZ_CAT = TWOMPZ_DIR + "results15_14_50_41_9.fits.gz"


def load_twompz(filename):
    """Load the 2MASS photoz catalog

    >>> ra, dec, z = load_twompz(TWOMPZ_CAT)
    >>> print ra[0], dec[0], z[0]
    3.81918741... rad 2.09295799... rad 0.0808...
    >>> print ra[-1], dec[-1], z[-1]
    5.53166782... rad 1.50311874... rad 0.0292...
    """
    gal_data = fits.open(filename)
    gal_dec = -gal_data[1].data['DEC'][:] + np.pi / 2.
    gal_ra = gal_data[1].data['RA'][:]
    gal_z = gal_data[1].data['ZPHOTO'][:]

    return gal_ra * u.rad, gal_dec * u.rad, gal_z


# -----------------------------------------------------------------------------
# MICE simulation catalogs
# use load_mice_cat()
# Note that this also returns the "value" field; which can be set to a galaxy
# property in the simulations.
# -----------------------------------------------------------------------------
def load_mice_cat(filename):
    """Load a re-processed MICE catalog"""
    gal_data = h5py.File(filename, "r")
    gal_dec = -gal_data['dec'].value * np.pi / 180. + np.pi / 2.
    gal_ra = gal_data['ra'].value * np.pi / 180.
    gal_z = gal_data['z'].value
    gal_value = gal_data['value'].value

    return gal_ra * u.rad, gal_dec * u.rad, gal_z, gal_value


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)

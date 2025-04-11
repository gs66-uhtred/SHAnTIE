import numpy as np
import ephem
import healpy
#import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import h5py
from numpy import linalg as LA
#rcdef = plt.rcParams.copy()
from astropy.io import fits

#root_data = '/Volumes/data_drive/data/'
#root_data = '/home/eswitzer/data/FIRAS/LAMBDA_fits/'
#root_data = '/Users/cjander8/Desktop/Kappa_backup/11_01_18/FIRAS_files/'
root_data = '/local//cjander8/data/FIRAS_files/'

def read_firas(low=False, gal_cut=10.):
    """galcut removes dipole; important toward lowf"""
    if low:
        #firas_file_fits = root_data + 'FIRAS/firas_hpx_destriped_sky_spectra_lowf_fixed.fits'
        #firas_weight_fits = root_data + 'FIRAS/firas_hpx_c_vector_lowf_fixed.fits'
        firas_file_fits = root_data + 'firas_hpx_destriped_sky_spectra_lowf_fixed.fits'
        firas_weight_fits = root_data + 'firas_hpx_c_vector_lowf_fixed.fits' 
        firas_file_fits = root_data + 'firas_hpx_destriped_sky_spectra_lowf_v2.fits'
        firas_weight_fits = root_data + 'firas_hpx_c_vector_lowf_v2.fits'
    else:
        #firas_file_fits = root_data + 'FIRAS/firas_hpx_destriped_sky_spectra_high_fixed.fits'
        #firas_weight_fits = root_data + 'FIRAS/firas_hpx_c_vector_high_fixed.fits'
        firas_file_fits = root_data + 'firas_hpx_destriped_sky_spectra_high_fixed.fits'
        firas_weight_fits = root_data + 'firas_hpx_c_vector_high_fixed.fits'
        firas_file_fits = root_data + 'firas_hpx_destriped_sky_spectra_high_v2.fits'
        firas_weight_fits = root_data + 'firas_hpx_c_vector_high_v2.fits'
    firas_data = fits.open(firas_file_fits)
    firas_angweight_raw = firas_data[1].data['WEIGHT']
    firas_spectrum = firas_data[1].data['SPECTRUM']

    firas_cvec_data = fits.open(firas_weight_fits)
    # noise in units of MJy/sr
    firas_cvec = firas_cvec_data[1].data['C_VECTOR'][0]

    firas_nnu = firas_spectrum.shape[1]
    if low:
        freq = np.arange(firas_nnu) * 13.604162 + 68.020812
    else:
        freq = np.arange(firas_nnu) * 13.604162 + 612.18729

    firas_ring = np.zeros_like(firas_spectrum)
    firas_wdipole = np.zeros_like(firas_spectrum)
    for ind in range(len(freq)):
        firas_wdipole[:, ind] = healpy.reorder(firas_spectrum[:, ind], n2r=True)
        if gal_cut:
            try:
                firas_ring[:, ind] = healpy.pixelfunc.remove_dipole(firas_wdipole[:, ind],
                                                                    nest=False,
                                                                    bad=0, gal_cut=gal_cut);
            except:
                firas_ring[:, ind] = np.zeros_like(firas_angweight_raw)
        else:
            firas_ring[:, ind] = np.copy(firas_wdipole[:, ind])

    firas_angweight = healpy.reorder(firas_angweight_raw, n2r=True)
    firas_nside = healpy.get_nside(firas_angweight)

    return firas_ring, freq, firas_angweight, firas_cvec, firas_nside

def define_z(freq, line="CII"):
    # line freqs in GHz
    nu0 = {"CII": 1897.,
           "CO21": 230.54,
           "CO10": 115.}

    #correct CII frequency
    nu0["CII"] = .299792/(157.7*10**-6)

    #Add fake dummy line.
    nu0["4000"] = 4000.

    nu0_line = nu0[line]

    # here, taking the freq at midpoint of the bin
    redshift_left = nu0_line / (freq + 13.604162/2.) - 1.
    redshift_right = nu0_line / (freq - 13.604162/2.) - 1.

    mask = redshift_left >= 0
    z_left = redshift_left[mask]
    z_right = redshift_right[mask]

    return z_left, z_right, mask

def bin_catalog(ra_rad, dec_rad, redshift, nside, z_left, z_right):
    npix = healpy.nside2npix(nside)
    nnu = len(z_left)
    # from Ra/dec to galactic
    rotate = healpy.rotator.Rotator(coord=['C','G'])
    theta_gal, phi_gal = rotate(dec_rad, ra_rad)

    gal_ind = healpy.pixelfunc.ang2pix(nside, theta_gal, phi_gal, nest=False)

    # spatial density
    gal_spatial = np.bincount(gal_ind, minlength=npix)

    # spectral binning
    gal_ring = np.zeros(shape=(npix, nnu))
    gal_counts = np.zeros_like(z_left)
    for ind in range(nnu):
        in_bin = np.logical_and(redshift > z_left[ind], redshift < z_right[ind])
        gal_bin = gal_ind[in_bin]
        gal_ring[:,ind] = np.bincount(gal_bin, minlength=npix)
        gal_counts[ind] = len(gal_bin)

    # make a separable selection function
    nbar = gal_spatial[:, None] * gal_counts[None, :]
    nbar *= np.sum(gal_counts) / np.sum(nbar)

    overdensity = (gal_ring - nbar) / nbar

    return overdensity, nbar, gal_counts, gal_spatial

def crosspower(im_map, im_weight, gal_map, gal_weight, modecut, ntrial=None):
    is_bad = np.logical_or(np.isnan(gal_map), np.isinf(gal_map))
    gal_map[is_bad] = 0.
    gal_weight[is_bad] = 0.
    im_map[is_bad] = 0.
    im_weight[is_bad] = 0.

    # weight both inputs
    gal_weighted = gal_map * gal_weight
    im_weighted = im_map * im_weight

    # remove the mean of both signals
    im_mean = np.mean(im_weighted, axis=1)
    im_weighted -= im_mean[:, None]

    gal_mean = np.mean(gal_weighted, axis=1)
    gal_weighted -= gal_mean[:, None]

    Uim, sim, Vim = np.linalg.svd(im_weighted, full_matrices=False)
    sim_cut = np.copy(sim)
    sim_cut[modecut] = 0
    im_cleaned = np.dot(np.dot(Uim, np.diag(sim_cut)), Vim)

    # null the same modes in the galaxy sample
    gal_fg_basis = np.dot(gal_weighted, Vim.T)
    gal_fg_basis[:, modecut] = 0.
    gal_cleaned = np.dot(gal_fg_basis, Vim)
    #gal_cleaned = np.copy(gal_weighted)

    # remove the mean from the cleaned maps (not needed?)
    im_mean = np.mean(im_cleaned, axis=1)
    im_cleaned -= im_mean[:, None]

    gal_mean = np.mean(gal_cleaned, axis=1)
    gal_cleaned -= gal_mean[:, None]

    cross = np.dot(gal_cleaned.flatten(), im_cleaned.flatten())
    cross /= np.dot(im_weight.flatten(), gal_weight.flatten())

    indices=np.arange(gal_weighted.shape[0])

    im_std = np.sqrt(1. / im_weight)

    if ntrial:
        result = np.zeros(ntrial)
        for trial in range(ntrial):
            np.random.shuffle(indices)
            gal_redraw = gal_cleaned[indices, :]
            gal_weight_redraw = gal_weight[indices, :]
            single = np.dot(gal_redraw.flatten(), im_cleaned.flatten())
            single /= np.dot(im_weight.flatten(), gal_weight_redraw.flatten())

            # gal_weight must be nbar, properly normalized to do this
            #randcat = np.random.poisson(lam=gal_weight)
            #randim = np.random.normal(size=im_cleaned.shape)
            #randim *= im_std
            #single = np.dot(randcat.flatten(), randim.flatten())
            #single /= np.dot(im_weight.flatten(), gal_weight.flatten())

            result[trial] = single
    else:
        result = None

    return cross, result, im_cleaned, gal_cleaned, Vim, sim

def boxcar_conv_eclipctic(map_in, fwhm = 2.4, phi_tol = 0.125, verbose = False, vectorized = True):
    #Convolve in theta direction with narrow boxcar of fwhm degrees.
    #Map should be in ecliptic coordinates to approximate FIRAS scan strategy.
    #Convolves slightly in phi direction, by maximum of fwhm*phi_tol.
    #Try a loop for now. Might be very slow.
    pix_indeces = np.arange(map_in.size)
    lon_lat = healpy.pix2ang(int((map_in.size/12)**0.5), pix_indeces, lonlat=True)
    map_out = np.zeros(map_in.shape)
    if not vectorized:
        for pix_num in pix_indeces:
            #if verbose:
            #    print pix_num
            lon_diff = np.abs((lon_lat[0] - lon_lat[0][pix_num])%360)
            lat_diff = np.abs(lon_lat[1] - lon_lat[1][pix_num])
            box_car_bool = np.logical_and(lon_diff <= fwhm*phi_tol/2., lat_diff <= fwhm/2.)
            map_out[pix_num] = np.mean(map_in[box_car_bool])
    else:
        lon_diff_tensor = np.abs((lon_lat[0][:,None] - lon_lat[0][None,:])%360)
        lat_diff_tensor = np.abs(lon_lat[1][:,None] - lon_lat[1][None,:])
        box_car_bool = np.logical_and(lon_diff_tensor <= fwhm*phi_tol/2., lat_diff_tensor <= fwhm/2.)
        box_car_bool = box_car_bool.astype(np.float)
        box_car_bool /= np.sum(box_car_bool, axis=1)[:,None]
        map_out = np.einsum('ij,j', box_car_bool, map_in)
    return map_out

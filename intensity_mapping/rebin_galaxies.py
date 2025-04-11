'''
Bin galaxy surveys onto healpix/redshift cubes
'''
# pylint: disable=no-member, no-name-in-module
import numpy as np
import healpy
import h5py
import astropy.units as u


def zbins_center(freq, nu0_line):
    """Find the redshift bins given a regular grid of central band frequencies
    This needs to infer a band width

    >>> freq = np.array([0.9, 1.0, 1.1])
    >>> z_l, z_r, mask = zbins_center(freq, 1.)
    >>> print z_l
    [ 0.05263158 -0.04761905 -0.13043478]
    >>> print z_r
    [ 0.17647059  0.05263158 -0.04761905]
    >>> print mask
    [ True False False]
    """
    delta_nu = freq[1] - freq[0]

    # here, taking the freq at midpoint of the bin
    z_left = nu0_line / (freq + delta_nu / 2.) - 1.
    z_right = nu0_line / (freq - delta_nu / 2.) - 1.

    mask = z_left >= 0

    return z_left, z_right, mask


def bin_catalog_z(redshift, z_left, z_right):
    """Count up the object in each redshift bin
    This is used to refine the redshift binning range of a survey
    e.g. z_left < z <= z_right

    >>> zs = np.array([0., 0.5, 1., 2.5])
    >>> zl = np.array([0., 1., 2.])
    >>> zr = np.array([1., 2., 3.])
    >>> print bin_catalog_z(zs, zl, zr)
    [ 2.  0.  1.]
    """
    num_z = len(z_left)
    gal_counts = np.zeros_like(z_left)

    for ind in range(num_z):
        in_bin = np.logical_and(redshift > z_left[ind],
                                redshift <= z_right[ind])

        gal_counts[ind] = np.sum(in_bin)

    return gal_counts


def bin_catalog2d(ra_rad, dec_rad, redshift, nside, z_left, z_right,
                  weights=None, coord="G", normalize=False):
    """Count up the number of objects in healpix pixels, for each redshift bin
    weights optionally weighs each entry by e.g. halo mass, luminosity, etc.

    Adds to bin if z_l < z_source <= z_r

    An example that puts 4 source across seceral redshifts:
    >>> zs = np.array([0.1, 0.2, 2., 3.5])
    >>> ra = np.array([0., 0.5, 1., 1.5]) * u.rad
    >>> dec = np.array([0., 0.5, 1., 1.5]) * u.rad
    >>> zl = np.array([0., 1., 2.])
    >>> zr = np.array([1., 2., 3.])
    >>> gal_counts, gal_cube = bin_catalog2d(ra, dec, zs, 16, zl, zr)
    >>> print gal_counts
    [ 2.  1.  0.]
    >>> print gal_cube.shape
    (3072, 3)
    >>> print np.where(gal_cube[:, 0] > 0)[0]
    [ 822 1527]
    >>> print np.where(gal_cube[:, 1] > 0)[0]
    [1981]
    >>> print np.where(gal_cube[:, 2] > 0)[0]
    []
    """
    npix = healpy.nside2npix(nside)
    num_z = len(z_left)

    # from Ra/dec to galactic
    rotate = healpy.rotator.Rotator(coord=['C', coord])
    theta_gal, phi_gal = rotate(dec_rad.to(u.rad).value,
                                ra_rad.to(u.rad).value)

    gal_ind = healpy.pixelfunc.ang2pix(nside, theta_gal, phi_gal, nest=False)

    # spatial density
    #gal_spatial = np.bincount(gal_ind, minlength=npix)

    # spectral binning
    gal_cube = np.zeros(shape=(npix, num_z))
    gal_counts = np.zeros_like(z_left)
    if normalize:
        counts_cube = np.zeros(gal_cube.shape)
    for ind in range(num_z):
        # restrict to the galaxies in this redshift bin
        in_bin = np.logical_and(redshift > z_left[ind],
                                redshift <= z_right[ind])

        gal_bin = gal_ind[in_bin]
        if weights is not None:
            weights_bin = weights[in_bin]
        else:
            weights_bin = None

        # sum them up on the shell and find the total counts
        gal_cube[:, ind] = np.bincount(gal_bin, minlength=npix,
                                       weights=weights_bin)
        if normalize:
            counts_cube[:, ind] = np.bincount(gal_bin, minlength=npix, weights=None)
            nonzero_bool = counts_cube[:, ind] != 0
            gal_cube[:, ind][nonzero_bool] /= counts_cube[:, ind][nonzero_bool]

        gal_counts[ind] = np.sum(in_bin)

    return gal_counts, gal_cube


def bin_catalog_h5(outfile, freq, nu0_line, gal_ra, gal_dec, gal_z,
                   wh_z_good=None, nside=128, counts_cut=0.01,
                   coord="G", weights=None):
    """Make a data cube hdf5 file given the nu0 line center, galaxy position
    catalog.

    Optionally restrict redshift range with wh_z_good. If not given explicitly
    then limit the range to where the counts per redshift slice fall below
    `counts_cut`, e.g. 1% of the maximum galaxy density
    """
    z_left, z_right, _ = zbins_center(freq, nu0_line)

    if wh_z_good is None:
        gal_counts = bin_catalog_z(gal_z, z_left, z_right)
        wh_z_good = gal_counts > counts_cut * np.max(gal_counts)

    gal_counts, gal_spatial = bin_catalog2d(gal_ra, gal_dec, gal_z, nside,
                                            z_left[wh_z_good],
                                            z_right[wh_z_good],
                                            coord=coord,
                                            weights=weights)

    fout = h5py.File(outfile, "w")
    _ = fout.create_dataset("datacube", compression="gzip",
                            compression_opts=9,
                            data=gal_spatial)

    fout["freq"] = freq[wh_z_good]
    fout["z_left"] = z_left[wh_z_good]
    fout["z_right"] = z_right[wh_z_good]
    fout["z_counts"] = gal_counts
    fout.close()

    return wh_z_good


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)

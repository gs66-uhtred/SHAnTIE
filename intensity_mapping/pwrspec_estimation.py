import numpy as np
import healpy as hp
from . import sphere_music as sm
from scipy import spatial
import astropy.units as u
from intensity_mapping import sphere_music as sm
from intensity_mapping import real_alm as ra
import healpy
import scipy.interpolate

def all_crosses(maps, weight_map, beam_correction, bandpower_definition,
                mixing_matrix=None, zero_mode=None, apply_beam_correction = True):
    """Given maps, weights, beam corrections and bandpower definitions,
    find the cross-power among all input maps.
    """
    npix = weight_map.shape[0]
    #mode_weight = float(np.sum(weight_map)) / float(npix)
    mode_weight = np.sum(weight_map ** 2.) ** 2.
    mode_weight /= np.sum(weight_map ** 4.) * float(npix)

    power_matrix = {}
    beamcorr_power_matrix = {}
    for k1, v1 in maps.items():
        for k2, v2 in maps.items():
            cross_power_name = "%sx%s" % (k1, k2)
            print("calculating: ", cross_power_name)
            output = master_pwrspec(v1, weight_map, v2, weight_map,
                                    bandpower_definition, convert_cl=False,
                                    mixing_matrix=mixing_matrix,
                                    zero_mode=zero_mode,
                                    remove_mean=True)

            binned_ell, cls, _ = output
            power_matrix[cross_power_name] = cls
            if apply_beam_correction:
                beam1 = beam_correction[k1]
                beam2 = beam_correction[k2]
                beamcorr_power_matrix[cross_power_name] = cls / beam1 / beam2
            else:
                beamcorr_power_matrix[cross_power_name] = cls

    gaussian_errors = {}
    band_norm = np.max(bandpower_definition, axis=1)
    delta_vec = np.sum(bandpower_definition / band_norm[:, None], axis=1)
    nmode = (2. * binned_ell + 1.) * mode_weight * delta_vec
    gerr_norm = 1. / nmode
    for k1, v1 in beamcorr_power_matrix.items():
        for k2, v2 in beamcorr_power_matrix.items():
            cov_entry = "[%s]x[%s]" % (k1, k2)
            ind_i, ind_j = k1.split("x")
            ind_k, ind_l = k2.split("x")
            #pair1 = "%sx%s" % (ind_i, ind_j)
            #pair2 = "%sx%s" % (ind_k, ind_l)
            pair1 = "%sx%s" % (ind_i, ind_l)
            pair2 = "%sx%s" % (ind_j, ind_k)
            pair3 = "%sx%s" % (ind_i, ind_k)
            pair4 = "%sx%s" % (ind_j, ind_l)
            var = beamcorr_power_matrix[pair1] * beamcorr_power_matrix[pair2]
            var += beamcorr_power_matrix[pair3] * beamcorr_power_matrix[pair4]
            gaussian_errors[cov_entry] = var * gerr_norm

    return power_matrix, beamcorr_power_matrix, gaussian_errors

def all_crosses_matrix(maps, weight_maps, beam_correction, bandpower_definition,
                mixing_matrix=None, zero_mode=None, apply_beam_correction = True, remove_mean = False, remove_weighted_mean = True, skip_redundancies = True, abs_val_cov = True):
    """Given maps, weights, beam corrections and bandpower definitions,
    find the cross-power among all input maps.
    """
    #maps should have shape (num to correlate, Nside). Weight_maps should be None or have shape (Nside) or shape (num to correlate, Nside)
    #beam_correction has shape (num to correlate, l_length). It is the 1-sided correction, or square-root of beam transfer function for identical beams.
    #Mixing matrix is either None, has shape (l_length, l_length), or has shape (num to correlate, num to correlate, l_length, l_length)
    npix = maps.shape[-1]
    if type(weight_maps) == type(None):
        mode_weights = 1.
        weight_bool = False
    elif len(weight_maps.shape) == 2:
        mode_weights = np.einsum('ij,kj',weight_maps,weight_maps) ** 2.
        mode_weights /= np.einsum('ij,kj',weight_maps**2.,weight_maps**2.) *float(npix)
        weight_bool = True
    elif len(weight_maps.shape) == 1:
        mode_weights = np.einsum('j,j',weight_maps,weight_maps) ** 2.
        mode_weights /= np.einsum('j,j',weight_maps**2.,weight_maps**2.) *float(npix)
        weight_bool = False

    if type(mixing_matrix) == type(None):
        mix_bool = False
        mix = None
    elif len(mixing_matrix.shape) ==2:
        mix_bool = False
        mix = mixing_matrix 
    else:
        mix_bool = True
    maps_len = maps.shape[0]
    bin_len = bandpower_definition.shape[0]
    power_matrix = np.zeros((bin_len, maps_len, maps_len))
    beamcorr_power_matrix = np.zeros((bin_len, maps_len, maps_len))
    for k1 in range(maps_len):
        v1 = maps[k1,:]
        if skip_redundancies:
            next_range = list(range(k1+1))
        else:
            next_range = list(range(maps_len))
        for k2 in next_range:
            v2 = maps[k2,:]
            cross_power_name = "%sx%s" % (k1, k2)
            if mix_bool == True:
                mix = mixing_matrix[k1,k2,:]
            if weight_bool == True:
                weight1 = weight_maps[k1,:]
                weight2 = weight_maps[k2,:]
            elif type(weight_maps) == type(None):
                weight1 = np.ones(v1.shape)
                weight2 = np.ones(v2.shape)
            else:
                weight1 = weight_maps
                weight2 = weight_maps
            print("calculating: ", cross_power_name)
            if not apply_beam_correction:
                output = master_pwrspec(v1, weight1, v2, weight2,
                                    bandpower_definition, convert_cl=False,
                                    mixing_matrix=mix,
                                    zero_mode=zero_mode, remove_mean=remove_mean,
                                    remove_weighted_mean=remove_weighted_mean)
            else:
                output = master_pwrspec(v1, weight1, v2, weight2,
                                    bandpower_definition, convert_cl=False,
                                    mixing_matrix=mix,
                                    zero_mode=zero_mode, remove_mean=remove_mean,
                                    remove_weighted_mean=remove_weighted_mean, tfunc = beam_correction[k1,:]*beam_correction[k2,:])

            binned_ell, cls, _ = output
            power_matrix[:,k1,k2] = cls
            beamcorr_power_matrix[:,k1,k2] = cls
            if skip_redundancies:
                power_matrix[:,k2,k1] = cls
                beamcorr_power_matrix[:,k2,k1] = cls

    band_norm = np.max(bandpower_definition, axis=1)
    delta_vec = np.sum(bandpower_definition / band_norm[:, None], axis=1)
    if type(mode_weights) == np.ndarray:
        nmode = (2. * binned_ell[:,None,None] + 1.) * delta_vec[:,None,None] * mode_weights[None,:,:]
        gerr_norm = 1. / nmode
        gaussian_errors = np.einsum('mil,mjk->mijkl', beamcorr_power_matrix*(gerr_norm**0.5), beamcorr_power_matrix*(gerr_norm**0.5))
        if abs_val_cov:
            gaussian_errors = np.abs(gaussian_errors)
        gaussian_errors_2nd_term = np.einsum('mik,mjl->mijkl', beamcorr_power_matrix*(gerr_norm**0.5), beamcorr_power_matrix*(gerr_norm**0.5))
        if abs_val_cov:
            gaussian_errors_2nd_term = np.abs(gaussian_errors_2nd_term)
        gaussian_errors += gaussian_errors_2nd_term
    else:
        nmode = (2. * binned_ell + 1.) * mode_weights * delta_vec
        gerr_norm = 1. / nmode
    
        gaussian_errors = np.einsum('mil,mjk->mijkl', beamcorr_power_matrix*(gerr_norm[:,None,None]**0.5), beamcorr_power_matrix*(gerr_norm[:,None,None]**0.5))
        if abs_val_cov:
            gaussian_errors = np.abs(gaussian_errors)
        gaussian_errors_2nd_term = np.einsum('mik,mjl->mijkl', beamcorr_power_matrix*(gerr_norm[:,None,None]**0.5), beamcorr_power_matrix*(gerr_norm[:,None,None]**0.5))
        if abs_val_cov:
            gaussian_errors_2nd_term = np.abs(gaussian_errors_2nd_term)
        gaussian_errors += gaussian_errors_2nd_term
    #gaussian_errors *= gerr_norm

    return power_matrix, beamcorr_power_matrix, gaussian_errors

def f_sky(weights):
    factor = np.sum(weights**2.)**2.
    factor /= np.sum(weights**4.)*float(np.size(weights))
    return factor

def gauss_cov_nmodes(clzz, ell_vec, weights = None, bin_size = 1.):
    #Full sky if no weights specified.
    cov = np.einsum('ijm,ikl->ijklm', clzz, clzz)
    cov += np.einsum('ijl,ikm->ijklm', clzz, clzz)
    print(cov.shape)
    print(ell_vec.shape)
    cov /= (2*ell_vec + 1)[:,None,None,None,None]*bin_size
    if type(weights)!=type(None):
        if(weights.ndim == 1):
            factor = np.sum(weights**2.)**2.
            factor /= np.sum(weights**4.)*float(np.size(weights))
            print(factor)
            cov /= factor
        elif weights.ndim == 2:
            factor = np.einsum('ij,kj', weights, weights)**2.
            factor /= np.einsum('ij, kj', weights**2., weights**2.)*float(np.shape(weights)[-1])
            cov = cov[None,None,:,:,:,:,:]/factor[:,:,None,None,None,None,None]
    return cov

def gauss_cov_nmodes_exact(clzz, bandpower_definition, weights = None):
    #Full sky if no weights specified.
    #clzz must be unbinned.
    #Bandpower definition is (bins,total_ell) matrix with 1's on included ells.
    cov = np.einsum('ijm,ikl->ijklm', clzz, clzz)
    cov += np.einsum('ijl,ikm->ijklm', clzz, clzz)
    ell_vec = np.arange(bandpower_definition.shape[-1])
    cov /= (2*ell_vec + 1)[:,None,None,None,None]
    cov = np.einsum('ijklm, ni->njklm',cov, bandpower_definition)/(np.sum(bandpower_definition, axis=-1)[:,None,None,None,None]**2.)
    if type(weights)!=type(None):
        if(weights.ndim == 1):
            factor = np.sum(weights**2.)**2.
            factor /= np.sum(weights**4.)*float(np.size(weights))
            print(factor)
            cov /= factor
        elif weights.ndim == 2:
            factor = np.einsum('ij,kj', weights, weights)**2.
            factor /= np.einsum('ij, kj', weights**2., weights**2.)*float(np.shape(weights)[-1])
            print(factor)
            cov = cov[None,None,:,:,:,:,:]/factor[:,:,None,None,None,None,None]
    return cov


def truncate_clmat_cov(clmat, cov):
    #clmat has trailing dimensions z and z'.
    #cov has trailing dimensions z, z', z'', z'''.
    #Returns clmat and cov, eliminating redundant permutations, flattend in z,z'.
    z_dim = clmat.shape[-1]
    tril_bool = np.tril(np.ones((z_dim,z_dim))) == 1
    #triu_bool = np.triu(np.ones((z_dim,z_dim))) == 1
    comb_z_dim = np.sum(tril_bool)
    clmat_bool = np.ones(clmat.shape,dtype=bool)*tril_bool
    cov_bool = np.ones(cov.shape,dtype=bool)*np.einsum('ij,kl->ijkl', tril_bool, tril_bool)
    clmat_newshape = list(clmat.shape[0:-2])
    clmat_newshape.append(comb_z_dim)
    clmat_ans = clmat[clmat_bool].reshape(clmat_newshape)
    cov_newshape = list(cov.shape[0:-4])
    cov_newshape.append(comb_z_dim)
    cov_newshape.append(comb_z_dim)
    cov_ans = cov[cov_bool].reshape(cov_newshape)
    return clmat_ans, cov_ans

def return_truncated_z_indices(orig_z):
    z_dim = len(orig_z)
    tril_bool = np.tril(np.ones((z_dim,z_dim))) == 1
    #triu_bool = np.triu(np.ones((z_dim,z_dim))) == 1
    comb_z_dim = np.sum(tril_bool)
    data_mesh = np.meshgrid(orig_z,orig_z)
    data_z_trunc = []
    data_z_trunc.append(data_mesh[0][tril_bool].reshape(comb_z_dim))
    data_z_trunc.append(data_mesh[1][tril_bool].reshape(comb_z_dim))
    cov_bool = np.einsum('ij,kl->ijkl', tril_bool, tril_bool)
    cov_mesh = np.meshgrid(orig_z,orig_z,orig_z,orig_z)
    cov_z_trunc = []
    for ind in range(4):
        cov_z_trunc.append(cov_mesh[ind][cov_bool].reshape((comb_z_dim,comb_z_dim)))
    return data_z_trunc, cov_z_trunc

def all_crosses_weight_dict(maps, weight_maps, beam_correction, bandpower_definition, 
                            mixing_matrix=None, only_same_freq_errors=True, apply_beam_correction = True):
    """Given maps, weights, beam corrections and bandpower definitions,
    find the cross-power among all input maps.
    """
    mode_weight_matrix = {}
    power_matrix = {}
    beamcorr_power_matrix = {}
    for k1, v1 in maps.items():
        for k2, v2 in maps.items():
            cross_power_name = "%sx%s" % (k1, k2)
            if mixing_matrix != None:
                this_mix = mixing_matrix[cross_power_name]
            else:
                this_mix = None
            weight_map1 = weight_maps[k1]
            weight_map2 = weight_maps[k2]
            print("calculating: ", cross_power_name)
            if apply_beam_correction != True:
                output = master_pwrspec(v1, weight_map1, v2, weight_map2,
                                    bandpower_definition, convert_cl=False,
                                    mixing_matrix=this_mix,
                                    remove_mean=True)
            else:
                output = master_pwrspec(v1, weight_map1, v2, weight_map2,
                                    bandpower_definition, convert_cl=False,
                                    mixing_matrix=this_mix,
                                    remove_mean=True, tfunc = beam_correction)

            npix = weight_map1.shape[0]
            mode_weight = np.sum(weight_map1*weight_map2) ** 2.
            mode_weight /= np.sum((weight_map1*weight_map2) ** 2.) *float(npix)
            mode_weight_matrix[cross_power_name] = mode_weight

            binned_ell, cls, _ = output
            power_matrix[cross_power_name] = cls
            #if apply_beam_correction:
            #    beam1 = beam_correction[k1]
            #    beam2 = beam_correction[k2]
            #    beamcorr_power_matrix[cross_power_name] = cls / beam1 / beam2
            #else:
            #    beamcorr_power_matrix[cross_power_name] = cls
            beamcorr_power_matrix[cross_power_name] = cls

    gaussian_errors = {}
    band_norm = np.max(bandpower_definition, axis=1)
    delta_vec = np.sum(bandpower_definition / band_norm[:, None], axis=1)
    #nmode = (2. * binned_ell + 1.) * mode_weight * delta_vec
    #gerr_norm = 1. / nmode
    new_keys = list(beamcorr_power_matrix.keys())
    if only_same_freq_errors:
        new_keys = []
        for key in list(beamcorr_power_matrix.keys()):
            freqs = [int(s) for s in key.split('_') if s.isdigit()]
            if freqs[0] == freqs[1]:
                new_keys.append(key)
    #for k1, v1 in beamcorr_power_matrix.iteritems():
    for k1 in new_keys:
        v1 = beamcorr_power_matrix[k1]
        #for k2, v2 in beamcorr_power_matrix.iteritems():
        for k2 in new_keys:
            v2 = beamcorr_power_matrix[k2]
            cov_entry = "[%s]x[%s]" % (k1, k2)
            cross_power_name = "%sx%s" % (k1, k2)
            ind_i, ind_j = k1.split("x")
            ind_k, ind_l = k2.split("x")
            mode_weight = mode_weight_matrix[k1]
            nmode = (2. * binned_ell + 1.) * mode_weight * delta_vec
            gerr_norm = 1. / nmode
            #pair1 = "%sx%s" % (ind_i, ind_j)
            #pair2 = "%sx%s" % (ind_k, ind_l)
            pair1 = "%sx%s" % (ind_i, ind_l)
            pair2 = "%sx%s" % (ind_j, ind_k)
            pair3 = "%sx%s" % (ind_i, ind_k)
            pair4 = "%sx%s" % (ind_j, ind_l)
            var = beamcorr_power_matrix[pair1] * beamcorr_power_matrix[pair2]
            var += beamcorr_power_matrix[pair3] * beamcorr_power_matrix[pair4]
            gaussian_errors[cov_entry] = var * gerr_norm

    return power_matrix, beamcorr_power_matrix, gaussian_errors

def pullen_cov(c_ab, c_cd, c_ac, c_bd, c_ad, c_cb,  weight_a, weight_b, weight_c, weight_d, bandpower_definition, ell_vec, mix_ab=None, mix_cd=None):
   #Implements eq. 9 of arxiv:1707.06172
   #c_ab and c_cd are the binned measured cross powers to calculate errors on.
   #Note that for galaxy-galaxy power spectra, Pullen suggests input PS should be 0 if z1 != z2.
   #So, if b=gal(z1) and d=gal(z2), then the input c_bd=0 if z1 != z2.
    if type(mix_ab) == type(None):
        mix_ab = sm.mixing_from_weight(weight_a, weight_b, lmax=int(np.max(ell_vec)), iter=200)
    if type(mix_cd) == type(None):
        mix_cd = sm.mixing_from_weight(weight_c, weight_d, lmax=int(np.max(ell_vec)), iter=200)
    w_ac = hp.map2alm(weight_a*weight_c)
    w_bd = hp.map2alm(weight_b*weight_d)
    quad_acbd = sm.quad_coupling_kernel(ell_vec, w_ac, w_bd)
    w_ad = hp.map2alm(weight_a*weight_d)
    w_bc = hp.map2alm(weight_b*weight_c)
    quad_adbc = sm.quad_coupling_kernel(ell_vec, w_ad, w_bc)
    binned_ell, binning, unbinning = bin_unbin(ell_vec, bandpower_definition,
                                               convert_cl=False)
    #For some reason interp1d with extrapolate is not working.
    #Use polynomial fit to PS instead.
    def polyfit(c_input, ells_orig, ells_new, deg = 10):
        coeffs = np.polyfit(ells_orig, c_input, deg)
        ans = np.sum(coeffs*np.array([ells_new**(deg-n) for n in range(deg+1)]).T, axis = -1)
        return ans

    c_ac = polyfit(c_ac, binned_ell, ell_vec)
    c_bd = polyfit(c_bd, binned_ell, ell_vec)
    c_ad = polyfit(c_ad, binned_ell, ell_vec)
    c_cb = polyfit(c_cb, binned_ell, ell_vec)
    bracketed_mat = quad_acbd * c_ac[:,None]*c_bd[None,:] + quad_adbc * c_ad[:,None]*c_cb[None,:]
    bracketed_mat /= 2.*ell_vec[None,:] + 1.
    binned_bracketed_mat = np.dot(np.dot(binning, bracketed_mat), binning.T)
    mix_ab_binned = np.dot(binning, np.dot(mix_ab, unbinning))
    mix_ab_inv = np.linalg.pinv(mix_ab_binned)
    mix_cd_binned  = np.dot(binning, np.dot(mix_cd, unbinning))
    mix_cd_inv = np.linalg.pinv(mix_cd_binned)
    cov = np.dot(np.dot(mix_ab_inv, binned_bracketed_mat), mix_cd_inv.T)
    return cov

def test_gaussian_error(beamcorr_power_matrix, k1, k2):
    v1 = beamcorr_power_matrix[k1]
    v2 = beamcorr_power_matrix[k2]
    cov_entry = "[%s]x[%s]" % (k1, k2)
    cross_power_name = "%sx%s" % (k1, k2)
    ind_i, ind_j = k1.split("x")
    ind_k, ind_l = k2.split("x")
    pair1 = "%sx%s" % (ind_i, ind_l)
    pair2 = "%sx%s" % (ind_j, ind_k)
    pair3 = "%sx%s" % (ind_i, ind_k)
    pair4 = "%sx%s" % (ind_j, ind_l)
    var = beamcorr_power_matrix[pair1] * beamcorr_power_matrix[pair2]
    var += beamcorr_power_matrix[pair3] * beamcorr_power_matrix[pair4]
    return var

def correlated_synfast(cla, clb, clx, nside=None, lmax=None):
    """Given the auto powers of maps a and b and the cross-power,
    Generate maps a and b which are drawn from that covariance

    The Cls must be reported l-by-l
    """
    ilen = ra.ilength(lmax, lmax)
    lm_index = np.arange(ilen)
    (lvec, mvec) = ra.i2lm(lmax, lm_index)

    alms_unity1 = np.zeros(ilen,'D')
    alms_unity1.real = np.random.standard_normal(ilen)
    alms_unity1.imag = np.random.standard_normal(ilen)

    alms_unity2 = np.zeros(ilen,'D')
    alms_unity2.real = np.random.standard_normal(ilen)
    alms_unity2.imag = np.random.standard_normal(ilen)

    # The inputs should follow a triangle inequality
    # but if these are determined empirically, that is not assured
    factor = cla - clx ** 2. / clb
    factor[factor < 0.] = 0.
    a1mult = np.sqrt(factor)

    factor = np.copy(clb)
    factor[factor < 0.] = 0.
    a2mult = clx / np.sqrt(factor)
    b2mult = np.sqrt(factor)

    a1mult[np.isnan(a1mult)] = 0.
    a2mult[np.isnan(a2mult)] = 0.
    b2mult[np.isnan(b2mult)] = 0.

    alms_a = a1mult[lvec] * alms_unity1 / np.sqrt(2)
    alms_a += a2mult[lvec] * alms_unity2 / np.sqrt(2)
    alms_b = b2mult[lvec] * alms_unity2 / np.sqrt(2)

    map_a = healpy.alm2map(alms_a, nside, lmax=lmax, pol=False, verbose=False)
    map_b = healpy.alm2map(alms_b, nside, lmax=lmax, pol=False, verbose=False)

    return map_a, map_b

def full_correlated_synfast(cl_cov_model, lmax=None):
    """Given a Cl model in (len(l), len(z*type), len(z*type)), realize a stack of alms.
    """
    if type(lmax) == type(None):
        lmax = cl_cov_model.shape[0] - 1
    model_shape = np.shape(cl_cov_model)
    ilen = ra.ilength(lmax,lmax)
    lm_index = np.arange(ilen)
    (lvec, mvec) = ra.i2lm(lmax, lm_index)

    dat_len = model_shape[1]
    alms = np.zeros((ilen, dat_len), 'D')
    m_zero_bool = mvec == 0

    ps_check = np.zeros((cl_cov_model.shape[0], dat_len, dat_len))

    mean = np.zeros(dat_len)

    #Cl[ell,:,:] must be symmetric positive semi-definite.
    for ell in np.arange(lmax + 1):
        lvec_bool = lvec == ell
        ml_zero_bool = np.logical_and(lvec_bool, m_zero_bool)
        alms.real[lvec_bool] = np.random.multivariate_normal(mean, 0.5*cl_cov_model[ell,:,:], size = len(lvec[lvec_bool]))
        alms.imag[lvec_bool] = np.random.multivariate_normal(mean, 0.5*cl_cov_model[ell,:,:], size = len(lvec[lvec_bool]))
        alms[ml_zero_bool] = np.sqrt(0.5)*np.random.multivariate_normal(mean, cl_cov_model[ell,:,:])
        #Check PS
        ps_check[ell,:,:] = np.einsum('ij,ik', alms[lvec_bool], np.conj(alms[lvec_bool]))/(ell + 0.5)
        alms[ml_zero_bool] *= np.sqrt(2.)
    return alms, ps_check

def compute_gauss_errors(ca_est, cb_est, cx_est, w_a, w_b, bandpower_definition, ell):
    #Let ca_est be the ca PS before unmixing is applied.
    #binned_ell, binning, unbinning = bin_unbin(ell, bandpower_definition,
    #                                           convert_cl=False)
    #binned_mixing = np.dot(binning, np.dot(mixing_matrix, unbinning))
    #mixinv = np.linalg.pinv(binned_mixing, rcond)
    band_norm = np.max(bandpower_definition, axis=1)
    delta_vec = np.sum(bandpower_definition / band_norm[:, None], axis=1)
    npix = w_a.shape[0]
    mode_weight_x = np.sum(w_a*w_b)**2.
    mode_weight_x /= np.sum((w_a*w_b)**2.) * float(npix)
    mode_weight_a = np.sum(w_a*w_a)**2.
    mode_weight_a /= np.sum((w_a*w_a)**2.) * float(npix)
    mode_weight_b = np.sum(w_b*w_b)**2.
    mode_weight_b /= np.sum((w_b*w_b)**2.) * float(npix)
    n_modes_x = (2.*ell + 1.)*mode_weight_x*delta_vec
    n_modes_a = (2.*ell + 1.)*mode_weight_a*delta_vec
    n_modes_b = (2.*ell + 1.)*mode_weight_b*delta_vec
    #cov_x = np.zeros((len(cx_est),len(cx_est))
    #cov_a = np.zeros((len(ca_est),len(ca_est))
    #cov_b = np.zeros((len(cb_est),len(cb_est))
    #cov_x = (cov_x+cx_est)*(cov_x+cx_est).T 
    var_x = (ca_est*cb_est + cx_est**2.)/n_modes_x
    var_a = 2*(ca_est**2.)/n_modes_a
    var_b = 2*(cb_est**2.)/n_modes_b
    return var_a, var_b, var_x

def mc_cross_draw(ca, cb, cx, bandpower_definition,
                    weight=None, weight_b=None, mixing_matrix=None, nside=512, lmax=None, n_sample=100, verbose = False, rcond = 1.*10**-15):
    """Take binned power spectra for axa (ca), bxb (cb) and cross axb (cx)

    TODO: allow independent left and right weights
    """
    small_ell = np.arange(bandpower_definition.shape[1])
    binned_ell, binning, unbinning = bin_unbin(small_ell,
                                               bandpower_definition)

    ubcla = np.dot(unbinning, ca)
    ubclb = np.dot(unbinning, cb)
    ubclx = np.dot(unbinning, cx)
    ub_ell_vec = np.arange(ubcla.shape[0])

    nbin = ca.shape[0]

    npix = healpy.nside2npix(nside)
    if lmax is None:
        lmax = 3 * nside - 1

    if weight is None:
        weight = np.ones(shape=npix)
    if weight_b is None:
        weight_b = weight

    if mixing_matrix is None:
        mixing_matrix['cx'] = sm.mixing_from_weight(weight, weight_b, lmax=lmax)
        mixing_matrix['ca'] = sm.mixing_from_weight(weight, weight, lmax=lmax)
        mixing_matrix['cb'] = sm.mixing_from_weight(weight_b, weight_b, lmax=lmax)

    if type(mixing_matrix) == np.ndarray:
        mix = np.copy(mixing_matrix)
        mixing_matrix = {}
        mixing_matrix['cx'] = mix
        mixing_matrix['ca'] = mix
        mixing_matrix['cb'] = mix

    sampled_ca = np.zeros(shape=(n_sample, nbin))
    sampled_cb = np.zeros(shape=(n_sample, nbin))
    sampled_cx = np.zeros(shape=(n_sample, nbin))

    for ind in range(n_sample):
        if verbose:
            print(ind)
        map_a, map_b = correlated_synfast(ubcla, ubclb, ubclx,
                                          nside=nside, lmax=lmax)

        output = master_pwrspec(map_a, weight, map_b, weight_b,
                                bandpower_definition, convert_cl=False,
                                mixing_matrix=mixing_matrix['cx'],
                                remove_mean=True, rcond=rcond)

        binned_ell, clx_realization, _ = output
        sampled_cx[ind, :] = clx_realization

        # measure a x a simulations
        output = master_pwrspec(map_a, weight, map_a, weight,
                                bandpower_definition, convert_cl=False,
                                mixing_matrix=mixing_matrix['ca'],
                                remove_mean=True, rcond=rcond)

        binned_ell, cla_realization, _ = output
        sampled_ca[ind, :] = cla_realization

        # measure b x b simulations
        output = master_pwrspec(map_b, weight_b, map_b, weight_b,
                                bandpower_definition, convert_cl=False,
                                mixing_matrix=mixing_matrix['cb'],
                                remove_mean=True, rcond=rcond)

        binned_ell, clb_realization, _ = output
        sampled_cb[ind, :] = clb_realization

    return binned_ell, sampled_ca, sampled_cb, sampled_cx


def master_auto(input_map, input_weight, mixing_matrix, lmax=None, rcond=10**-5.):
    ps_cls = hp.sphtfunc.anafast(input_map * input_weight, lmax=lmax)
    mixinv = np.linalg.pinv(mixing_matrix, rcond=rcond)
    #return np.dot(mixinv.T, ps_cls)
    return np.dot(mixinv, ps_cls)

def master_auto_test_from_mem(cltt, weight, mixing_matrix, n_avgs = 1, rcond=10**-15.):
    """run a realization of Cl PS through MASTER with weight
    """
    nside = hp.get_nside(weight)
    
    cmb_temp_map = hp.sphtfunc.synfast(cltt, nside, verbose=False)
    cltt_est = master_auto(cmb_temp_map, weight, mixing_matrix,rcond=rcond)
    cltt_pseudo = hp.sphtfunc.anafast(cmb_temp_map * weight)
    cltt_true = hp.sphtfunc.anafast(cmb_temp_map)
    ell = np.arange(len(cltt_est))

    if n_avgs>1:
        for ind in range(n_avgs-1):
            cmb_temp_map = hp.sphtfunc.synfast(cltt, nside, verbose=False)
            cltt_est += master_auto(cmb_temp_map, weight, mixing_matrix,rcond=rcond)
            cltt_pseudo += hp.sphtfunc.anafast(cmb_temp_map * weight)
            cltt_true += hp.sphtfunc.anafast(cmb_temp_map)
    cltt_est /= float(n_avgs)
    cltt_pseudo /= float(n_avgs)
    cltt_true /= float(n_avgs)

    return {"ell":ell, \
            "cltt_est":cltt_est, \
            "cltt_pseudo": cltt_pseudo, \
            "cltt_true": cltt_true}

def master_auto_test(cltt, weightfile, mixingfile):
    """run a realization of the CMB through MASTER with mask
    """
    mask_map = hp.read_map(weightfile)
    mixing_matrix = np.load(mixingfile)

    nside = hp.get_nside(mask_map)

    cmb_temp_map = hp.sphtfunc.synfast(cltt, nside)
    cltt_est = master_auto(cmb_temp_map, mask_map, mixing_matrix)
    cltt_pseudo = hp.sphtfunc.anafast(cmb_temp_map * mask_map)
    cltt_true = hp.sphtfunc.anafast(cmb_temp_map)
    ell = np.arange(len(cltt_est))

    return {"ell":ell, \
            "cltt_est":cltt_est, \
            "cltt_pseudo": cltt_pseudo, \
            "cltt_true": cltt_true}


def execute_master_auto_test():
    datapath = "/Users/eswitzer/data/WMAP/"
    weightfile = datapath + \
                 "wmap_ext_temperature_analysis_mask_r10_7yr_v4_nside256.fits"

    mixingfile = "master_auto_test_mixing.npy"
    #sm.master_auto_test_prep(weightfile, mixingfile)

    (_, cltt) = qs.cltt()
    ok = master_auto_test(cltt, weightfile, mixingfile)
    ok = master_auto_test(cltt, weightfile, mixingfile)


def distance_to_mask(mask):
    """To apodize a mask region, determine the distance from an unmasked
    point to the nearest masked region using a kdtree.
    """
    npix = mask.shape[0]
    nside = hp.npix2nside(npix)

    # work on a x,y,z rather than RA/Dec because the kdtree metric will fail on the sphere
    x_pix, y_pix, z_pix = hp.pix2vec(nside, np.arange(npix))

    wh_masked = mask == 0.
    wh_unmasked = mask > 0.

    vec_masked = np.stack([x_pix[wh_masked], y_pix[wh_masked], z_pix[wh_masked]])
    vec_unmasked = np.stack([x_pix[wh_unmasked], y_pix[wh_unmasked], z_pix[wh_unmasked]])

    tree = spatial.KDTree(vec_masked.T)
    distances = tree.query(vec_unmasked.T)

    distance_to_mask = np.zeros(shape=npix)
    distance_to_mask[wh_unmasked] = distances[0]

    return distance_to_mask


def poor_apodize(mask, smoothing_scale):
    """Cheap apodization of a mask
    """
    apodized = hp.sphtfunc.smoothing(mask, fwhm=smoothing_scale.to(u.rad).value)
    # now for a mask, ensure it goes to zero
    apodized *= mask
    # rescale between 0 and 1
    print(np.min(apodized[mask]))
    apodized[mask] -= np.min(apodized[mask])
    apodized /= np.max(apodized)

    return apodized


def make_regular_bins(ell_vec, step, startl=None, fill_irregular_ends = False):
    """
    Make a bin definition matrix for bins of regular size
    If the ell vec is not divisible by step, then the remaining multipoles
    are ignored.

    >>> ell_vec = np.arange(12)
    >>> bm = make_regular_bins(ell_vec, 5)
    >>> print bm
    [[ 1.  1.  1.  1.  1.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  0.  0.]]

    >>> ell_vec = np.arange(10)
    >>> bm = make_regular_bins(ell_vec, 5)
    >>> print bm
    [[ 1.  1.  1.  1.  1.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  1.  1.  1.  1.  1.]]
    """
    nell = ell_vec.shape[0]
    if startl is None:
        startl = np.min(ell_vec)

    lmax = np.max(ell_vec)

    lims = np.arange(startl, lmax + 2, step)
    l_left = lims[0:-1]
    l_right = lims[1:]

    nband = l_left.shape[0]

    binning_mat = np.zeros(shape=(nband, nell))
    for ind in range(nband):
        binning_mat[ind, l_left[ind]:l_right[ind]] = 1.

    if fill_irregular_ends:
        check_bin_bool = np.sum(binning_mat, axis = 0) == 0
        print(check_bin_bool)
        num_extra_bands = check_bin_bool[0] + check_bin_bool[-1]
        new_binning_mat = np.zeros(shape=(nband + num_extra_bands, nell))
        if check_bin_bool[0]:
            new_binning_mat[0,0:startl] = 1.
            new_binning_mat[1:nband+1,:] = binning_mat
        else:
            new_binning_mat[0:nband,:] = binning_mat
        if check_bin_bool[1]:
            new_binning_mat[-1,l_right[-1]:] = 1.
        binning_mat = new_binning_mat
    return binning_mat


def bin_unbin(ell_vec, binning_mat, convert_cl=False):
    """
    Make binning and unbinning matrices

    >>> lmax = 200
    >>> ell_vec = np.arange(lmax)
    >>> cls = ell_vec ** 2.
    >>> bd = make_regular_bins(ell_vec, 40)
    >>> bl, bm, um = bin_unbin(ell_vec, bd)
    >>> bc = np.dot(bm, cls)
    >>> ubc = np.dot(um, bc)
    >>> print bl
    [  19.5   59.5   99.5  139.5  179.5]
    >>> print bc
    [   513.5   3673.5  10033.5  19593.5  32353.5]
    >>> print ubc[::40]
    [   513.5   3673.5  10033.5  19593.5  32353.5]
    """
    binning = np.copy(binning_mat)
    unbinning = np.copy(binning.T)

    if convert_cl:
        ellnorm = ell_vec * (ell_vec + 1) / (2. * np.pi)
        binning *= ellnorm[None, :]
        unbinning /= ellnorm[:, None]

    norm = np.sum(binning_mat, axis=1)
    binning /= norm[:, None]

    binned_ell = np.dot(binning, ell_vec) / np.sum(binning, axis=1)

    return binned_ell, binning, unbinning


def bin_power(ell_vec, cls, deltal, warn=False, ell_axis=0):
    """
    Bin both the ell and cl vectors into delta l bins
    This uses direct numpy means rather the matrix operations

    If binning 11 samples by 5 bins this will throw away the last sample
    To be warned when this happens, choose warn=True

    Bin a multi-dimensional cl along identified `ell_axis`

    >>> lmax = 201
    >>> ell_vec = np.arange(lmax)
    >>> cls = ell_vec ** 2.
    >>> bl, bc = bin_power(ell_vec, cls, 40)
    >>> print bl
    [  19.5   59.5   99.5  139.5  179.5]
    >>> print bc
    [   513.5   3673.5  10033.5  19593.5  32353.5]

    >>> cls = np.ones((lmax, 5, 3))
    >>> cls *= np.arange(3)[None, None, :]
    >>> bl, bc = bin_power(ell_vec, cls, 40, ell_axis=0)
    >>> print bc[:, 4, 0]
    [ 0.  0.  0.  0.  0.]
    >>> print bc[:, 4, 1]
    [ 1.  1.  1.  1.  1.]
    >>> print bc[:, 4, 2]
    [ 2.  2.  2.  2.  2.]

    >>> cls = np.ones((5, lmax, 3))
    >>> cls *= np.arange(3)[None, None, :]
    >>> bl, bc = bin_power(ell_vec, cls, 40, ell_axis=1)
    >>> print bc[4, :, 0]
    [ 0.  0.  0.  0.  0.]
    >>> print bc[4, :, 1]
    [ 1.  1.  1.  1.  1.]
    >>> print bc[4, :, 2]
    [ 2.  2.  2.  2.  2.]
    """
    ellf = ell_vec.astype(float)

    # if not an even multiple of deltal bins, throw away
    newlen = ell_vec.size
    if (ell_vec.size % deltal) != 0:
        shorten_by = ell_vec.size % deltal
        if warn:
            print("WARNING: binning %d by %d throws out the last %d" % \
                   (newlen, deltal, shorten_by))

        newlen -= shorten_by

    num_bins = int(newlen / deltal)
    cls_shape = list(cls.shape)
    cls_shape[ell_axis] = num_bins
    cls_shape.insert(ell_axis+1, deltal)

    # truncate the array along the ell-axis (roll it to axis 0 then back)
    # see SO numpy-take-cant-index-using-slice
    truncated_cls = np.rollaxis(np.rollaxis(cls, ell_axis, 0)[0:newlen],
                                0, ell_axis + 1)

    binned_cls = truncated_cls.reshape(cls_shape).mean(axis=ell_axis + 1)
    binned_ell = ellf[0: newlen].reshape((num_bins, deltal)).mean(axis=1)

    return (binned_ell, binned_cls)


def master_pwrspec(map_left, weight_left, map_right, weight_right,
                   bandpower_definitions, convert_cl=False, mixing_matrix=None,
                   remove_mean=False, remove_weighted_mean=False, zero_mode=None, rcond=1.*10**-15, tfunc = None, zero_low_ell = False):
    """
    Estimate the cross-power using master
    generally try to use a pre-calculated mixing matrix

    TODO:
    norm applies an ell (ell + 1)/2 norm before binning in l
    """
    #lmax = bandpower_definitions.shape[1]
    lmax = bandpower_definitions.shape[1] - 1
    #ell_vec = np.arange(lmax)
    ell_vec = np.arange(lmax + 1)

    if mixing_matrix is None:
        mixing_matrix = sm.mixing_from_weight(weight_left, weight_right, lmax=lmax)

    binned_ell, binning, unbinning = bin_unbin(ell_vec, bandpower_definitions,
                                               convert_cl=convert_cl)

    # moved the transpose from bcross_est_cls line here
    #binned_mixing = np.dot(binning, np.dot(mixing_matrix.T, unbinning))
    if type(tfunc) == type(None):
        binned_mixing = np.dot(binning, np.dot(mixing_matrix, unbinning))
    else:
        binned_mixing = np.dot(binning, np.dot(mixing_matrix, tfunc[:,None]*unbinning))
    #binned_mixing = np.dot(unbinning, np.dot(mixing_matrix, binning))
    mixinv = np.linalg.pinv(binned_mixing, rcond)
    #mixinv = np.dot(binning, np.dot(np.linalg.pinv(mixing_matrix), unbinning))

    #Try removing mean before weighting. Only consider non-zero weight areas.
    if remove_mean:
        map_left -= np.mean(map_left[weight_left!=0])
        map_right -= np.mean(map_right[weight_right!=0])

    #Remove unbinned ell vectors from the unweighted maps before analysis.
    #This could effect higher ell as well if the map is not band-limited.
    if zero_low_ell:
        alm_left = hp.sphtfunc.map2alm(map_left)
        alm_right = hp.sphtfunc.map2alm(map_right)
        ilen = ra.ilength(lmax,lmax)
        lm_index = np.arange(ilen)
        (lvec, mvec) = ra.i2lm(lmax, lm_index)
        zero_ell_bool = np.sum(binning, axis = 0) == 0
        ell_zeros = ell_vec[zero_ell_bool]
        alm_bool = np.any(lvec[:,None]*np.ones(ell_zeros.shape)[None,:] == ell_zeros[None,:]*np.ones(lvec.shape)[:,None], axis = 1)
        #print alm_bool.shape
        #print alm_bool
        #print lvec[alm_bool]
        alm_left[alm_bool] = 0
        alm_right[alm_bool] = 0
        nside = int((np.size(map_left)/12)**0.5)
        map_left = hp.sphtfunc.alm2map(alm_left, nside)
        map_right = hp.sphtfunc.alm2map(alm_right, nside)

    wmap_left = np.copy(map_left)
    wmap_right = np.copy(map_right)
    if remove_weighted_mean:
        #wmap_left -= np.mean(weight_left * map_left)
        #wmap_right -= np.mean(weight_right * map_right)
        wmap_left -= np.sum(weight_left * map_left) / np.sum(weight_left)
        wmap_right -= np.sum(weight_right * map_right) / np.sum(weight_right)

    wmap_left *= weight_left
    wmap_right *= weight_right

    cross_pseudo_cls = hp.sphtfunc.anafast(wmap_left, wmap_right, lmax=lmax)
    #if zero_low_ell:
    #    zero_ell_bool = np.sum(binning, axis = 0) == 0
    #    cross_pseudo_cls[zero_ell_bool] = 0
    #bcross_pseudo_cls = np.dot(binning, cross_pseudo_cls[:-1])
    bcross_pseudo_cls = np.dot(binning, cross_pseudo_cls)

    # anafast is inclusive of lmax but the convention elsewhere is arange(lmax)
    #bcross_est_cls = np.dot(mixinv.T, bcross_pseudo_cls)

    if zero_mode is not None:
        bcross_pseudo_cls[0:zero_mode] = 0.

    bcross_est_cls = np.dot(mixinv, bcross_pseudo_cls)

    return binned_ell, bcross_est_cls, bcross_pseudo_cls


def estimate_power_alt(map_left, weight_left, map_right, weight_right,
                       bandpower_definitions, convert_cl=False, mixing_matrix=None,
                       remove_mean=False):
    """
    This is an alterate version of estimate_power that multiplies by M^-1
    before binning; but this often has conditioning problems

    Estimate the cross-power using master
    convert_cl applies an ell (ell + 1)/2 norm before binning in l
    can accept pre-calculated mixing matrix
    """
    lmax = bandpower_definitions.shape[1]
    ell_vec = np.arange(lmax)

    if mixing_matrix is None:
        mixing_matrix = sm.mixing_from_weight(weight_left, weight_right, lmax=lmax)

    mixinv = np.linalg.pinv(mixing_matrix)

    binned_ell, binning, unbinning = bin_unbin(ell_vec, bandpower_definitions,
                                               convert_cl=convert_cl)

    wmap_left = np.copy(map_left)
    wmap_right = np.copy(map_right)
    if remove_mean:
        wmap_left -= np.sum(weight_left * map_left) / np.sum(weight_left)
        wmap_right -= np.sum(weight_right * map_right) / np.sum(weight_right)

    wmap_left *= weight_left
    wmap_right *= weight_right

    cross_pseudo_cls = hp.sphtfunc.anafast(wmap_left, wmap_right, lmax=lmax)

    # anafast is inclusive of lmax but the convention elsewhere is arange(lmax)
    cross_est_cls = np.dot(mixinv.T, cross_pseudo_cls[:-1])

    # find the binned power spectra
    bcross_est_cls = np.dot(binning, cross_est_cls)
    bcross_pseudo_cls = np.dot(binning, cross_pseudo_cls[:-1])

    return binned_ell, bcross_est_cls, bcross_pseudo_cls


#------------------------------------------------------------------------------
# vestigial functions that inverted the l-by-l (unbinned mixing matrix)
#------------------------------------------------------------------------------
def estimate_power(map_left, weight_left, map_right, weight_right,
                   lmax=399, deltal=40, norm=False, mixing_matrix=None,
                   remove_mean=False):
    """
    This is an alterate version of estimate_power that multiplies by M^-1
    before binning; but this often has conditioning problems

    Estimate the cross-power using master
    norm applies an ell (ell + 1)/2 norm before binning in l
    can accept pre-calculated mixing matrix
    """
    if mixing_matrix is None:
        mixing_matrix = sm.mixing_from_weight(weight_left, weight_right, lmax=lmax)

    #several options for the mixing inverse (conditioning)
    mixinv = np.linalg.pinv(mixing_matrix)
    #mixinv = sm.eigh_inverse(mixing_matrix, cond_limit=0.00001)[1]
    #mixinv = np.linalg.inv(mixing_matrix)

    wmap_left = np.copy(map_left)
    wmap_right = np.copy(map_right)
    if remove_mean:
        wmap_left -= np.sum(weight_left * map_left) / np.sum(weight_left)
        wmap_right -= np.sum(weight_right * map_right) / np.sum(weight_right)

    wmap_left *= weight_left
    wmap_right *= weight_right

    cross_pseudo_cls = hp.sphtfunc.anafast(wmap_left, wmap_right, lmax=lmax)

    cross_est_cls = np.dot(mixinv.T, cross_pseudo_cls)

    ell_vec = np.arange(len(cross_pseudo_cls))

    if norm:
        norm = ell_vec * (ell_vec + 1) / 2. / np.pi
        cross_pseudo_cls *= norm
        cross_est_cls *= norm

    # find the binned power spectra
    (binned_ell, bcross_est_cls) = bin_power(ell_vec, cross_est_cls, deltal)
    (binned_ell, bcross_pseudo_cls) = bin_power(ell_vec, cross_pseudo_cls, deltal)

    return binned_ell, bcross_est_cls, bcross_pseudo_cls

def flat_sky_lvects(nx,ny,lx,ly, radians=False):
    #Returns all the lx and ly vectors and all the l_total vectors
    #nx and ny are the number of pixels in the x,y direction.
    #lx and ly are the angular sizes of the patch in the x and y direction.
    if not radians:
        #Convert from degrees to radians.
        lx *= np.pi/180.
        ly *= np.pi/180.
    dlx = 2.*np.pi/lx
    dly = 2.*np.pi/ly
    xn = np.arange(nx) - nx/2
    yn = np.arange(ny) - ny/2
    lx = dlx*xn
    ly = dly*yn
    l_tot = (lx[:,None]**2 + ly[None,:]**2)**0.5
    return lx, ly, l_tot
        
def flat_sky_modes_per_bin(l_left, l_right, nx, ny, lx, ly, radians=False):
    #Returns the number of modes per l_bin.
    l_tot = flat_sky_lvects(nx,ny,lx,ly, radians=radians)[2][None,:,:]*np.ones(l_left.size)[:,None,None]
    bin_bool = np.logical_and(l_tot>l_left[:,None,None], l_tot<=l_right[:,None,None])
    return np.sum(np.sum(bin_bool,axis=-1),axis=-1)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
    #execute_master_auto_test()

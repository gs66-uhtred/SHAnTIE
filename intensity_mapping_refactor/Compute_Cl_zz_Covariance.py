import numpy as np
import scipy
import pymaster as nmt
import healpy as hp
from numba import njit

def Tristram_approx_Nmt_single_z_bin(nmt_covariance_workspace, C_l_ac, C_l_bd, C_l_ad, C_l_bc, nmt_workspace, nmt_binning):
    return nmt.gaussian_covariance(nmt_covariance_workspace, 0, 0, 0, 0, [C_l_ac], [C_l_ad], [C_l_bc], [C_l_bd], nmt_workspace, wb=nmt_workspace)

def Tristram_approx_Nmt_z_by_z_auto(nmt_covariance_workspace, C_l_zz, nmt_workspace, nmt_binning, upper_diag_only = True):
    ell_size = nmt_binning.get_n_bands()
    z1_size = C_l_zz.shape[1]
    z2_size = C_l_zz.shape[2]
    cov = np.zeros((ell_size, z1_size, z2_size, ell_size, z1_size, z2_size))
    if upper_diag_only:
       print("Filling in upper diagonal only")
       indeces = np.triu_indices(z1_size,m=z2_size)
       #print(indeces)
       for z1, z2 in zip(indeces[0], indeces[1]):
           #print(z1)
           for z3, z4 in zip(indeces[0], indeces[1]):
               cov[:,z1,z2,:,z3,z4] = Tristram_approx_Nmt_single_z_bin(nmt_covariance_workspace, C_l_zz[:,z1,z3], C_l_zz[:,z2,z4], C_l_zz[:,z1,z4], C_l_zz[:,z2,z3], nmt_workspace, nmt_binning)
    return cov
                       
def simulate_1d_maps(cl_model, n_realizations, nside):
    realized_maps = np.empty((n_realizations, 12*nside**2))
    for ind in range(n_realizations):
        realized_maps[ind,:] = hp.synfast(cl_model, nside)
    return realized_maps

def cov_1d_from_map_draws(tom_pair, map_draw_list, separate_terms = False):
    if len(map_draw_list) == 1 and not separate_terms:
        maps = map_draw_list[0]
        n_realizations = maps.shape[0]
        cl_array = np.zeros((tom_pair.cl_zz.shape[0], n_realizations))
        for ind in range(n_realizations):
            cl_array[:,ind] = compute_cl_from_drawn_map_pair(tom_pair, maps[ind,:][np.newaxis,:], maps[ind,:][np.newaxis,:])
        cov = np.cov(cl_array)
        return cov
    if len(map_draw_list) == 1 and separate_terms:
        maps = map_draw_list[0]
        n_realizations = maps.shape[0]
        cl_array1 = np.zeros((tom_pair.cl_zz.shape[0], n_realizations))
        #cl_array2 = np.zeros((tom_pair.cl_zz.shape[0], n_realizations))
        for ind in range(n_realizations):
            cl_array1[:,ind] = compute_cl_from_drawn_map_pair(tom_pair, maps[ind,:][np.newaxis,:], maps[-(ind+1),:][np.newaxis,:])
        cov1 = np.cov(cl_array1)
        cov2 = np.cov(cl_array1, np.flip(cl_array1, axis=1))
        return [cov1, cov2]

def Simulate_cov_single_z_bin(tom_pair, C_l_list, n_realizations_multiplier = 0.25, auto_power = True):
    #tom_pair should be 1d. Will have data written to it. 
    n_realizations = int(n_realizations_multiplier*C_l_list[0].size)
    if len(C_l_list) == 1:
        #Correlate single C_l with itself.
        cl_array = np.zeros((tom_pair.cl_zz.shape[0], n_realizations))
        for ind in range(n_realizations):
            map_draws = hp.synfast(C_l_list[0], tom_pair.maps1.nside)[np.newaxis,:]
            cl_array[:,ind] = compute_cl_from_drawn_map_pair(tom_pair, map_draws, map_draws)
        cov = np.cov(cl_array)
        return cov
    if len(C_l_list) == 2:
        #compute Expectation_Value[alm_1*alm_2*alm_2*alm_1] as first term.
        #compute Expectation_Value[alm_1*alm_2*alm_1*alm_2] as second term.
        if auto_power == True:
            for ind in range(n_realizations):
                cl_array = np.zeros((tom_pair.cl_zz.shape[0], n_realizations)) 
                map_draw1 = hp.synfast(C_l_list[0], tom_pair.maps1.nside)[np.newaxis,:]
                map_draw2 = hp.synfast(C_l_list[1], tom_pair.maps1.nside)[np.newaxis,:]
                cl_array[:,ind] = compute_cl_from_drawn_map_pair(tom_pair, map_draw1, map_draw2)
                cov = np.cov(cl_array)
            return [cov, cov]
            

def compute_cl_from_drawn_map_pair(tom_pair, map1, map2):
    tom_pair.maps1.map_array = map1
    tom_pair.maps2.map_array = map2
    tom_pair.maps1.make_map_dict()
    tom_pair.maps2.make_map_dict()
    tom_pair.compute_cl_zz(bin_size = tom_pair.bins.get_nell_list()[0])
    return tom_pair.cl_zz[:,0,0]

def Tristram_approx_Nmt_separable_clzz_2components_sym_update_cov(nmt_covariance_workspace, C_l_i, C_l_j, C_zz_i, C_zz_j, nmt_workspace, nmt_binning, cov_so_far, cov_temp):
    cov_ll = Tristram_approx_Nmt_single_z_bin(nmt_covariance_workspace, C_l_i, C_l_j, C_l_i, C_l_j, nmt_workspace, nmt_binning)
    #cov_temp *= 0.5*cov_ll[:,None,None,:,None,None]*C_zz_i[None,:,None,None,:,None]*C_zz_j[None,None,:,None,None,:]
    #cov_so_far += cov_temp
    #cov_temp[:]=1.
    #cov_temp *= 0.5*cov_ll[:,None,None,:,None,None]*C_zz_i[None,:,None,None,None,:]*C_zz_j[None,None,:,None,:,None]
    #cov_so_far += cov_temp
    update_cov_loop_numba(0.5*cov_ll, C_zz_i, C_zz_j, cov_so_far)

@njit
def update_cov_loop_numba(cov_ll, zz_i, zz_j, cov_so_far):
    n_ell = cov_ll.shape[0]
    n_z1 = zz_i.shape[0]
    n_z2 = zz_i.shape[1]
    for ind1 in range(n_ell):
        for ind2 in range(n_z1):
            for ind3 in range(ind2,n_z2):
                for ind4 in range(n_ell):
                    for ind5 in range(n_z1):
                        for ind6 in range(ind5,n_z2):
                            cov_so_far[ind1,ind2,ind3,ind4,ind5,ind6] += cov_ll[ind1,ind4]*zz_i[ind2,ind5]*zz_j[ind3,ind6]
                            cov_so_far[ind1,ind2,ind3,ind4,ind5,ind6] += cov_ll[ind1,ind4]*zz_i[ind2,ind6]*zz_j[ind3,ind5]

def Tristram_approx_Nmt_separable_clzz_2components_term_acbd(nmt_covariance_workspace, C_l_i, C_l_j, C_zz_i, C_zz_j, nmt_workspace, nmt_binning):
    zeros = np.zeros_like(C_l_i)
    cov_ll_term1 = Tristram_approx_Nmt_single_z_bin(nmt_covariance_workspace, zeros, zeros, C_l_i, C_l_j, nmt_workspace, nmt_binning)
    ell_size = cov_ll_term1.shape[0]
    cov_lzzlzz_term1 = np.ones((ell_size, C_zz_i.shape[0], C_zz_i.shape[1], ell_size, C_zz_j.shape[0], C_zz_j.shape[1]))
    cov_lzzlzz_term1 *= cov_ll_term1[:,None,None,:,None,None]
    zzzz_term = np.ones((C_zz_i.shape[0], C_zz_i.shape[1], C_zz_j.shape[0], C_zz_j.shape[1]))
    zzzz_term *= C_zz_i[:,:,None,None]
    zzzz_term *= C_zz_j[None,None,:,:]
    zzzz_term = np.einsum('jmkn->jkmn', zzzz_term)
    cov_lzzlzz_term1 *= zzzz_term[None,:,:,None,:,:]
    #cov_lzzlzz_term1 = np.einsum('ijmlkn->ijklmn',cov_ll_term1[:,None,None,:,None,None]*C_zz_i[None,:,:,None,None,None]*C_zz_j[None,None,None,None,:,:])
    return cov_lzzlzz_term1

def Tristram_approx_Nmt_separable_clzz_2components_term_adbc(nmt_covariance_workspace, C_l_i, C_l_j, C_zz_i, C_zz_j, nmt_workspace, nmt_binning):
    zeros = np.zeros_like(C_l_i)
    cov_ll_term2 = Tristram_approx_Nmt_single_z_bin(nmt_covariance_workspace, C_l_i, C_l_j, zeros, zeros, nmt_workspace, nmt_binning)
    ell_size = cov_ll_term2.shape[0]
    cov_lzzlzz_term2 = np.ones((ell_size, C_zz_i.shape[0], C_zz_i.shape[1], ell_size, C_zz_j.shape[0], C_zz_j.shape[1]))
    cov_lzzlzz_term2 *= cov_ll_term2[:,None,None,:,None,None]
    zzzz_term = np.ones((C_zz_i.shape[0], C_zz_i.shape[1], C_zz_j.shape[0], C_zz_j.shape[1]))
    zzzz_term *= C_zz_i[:,:,None,None]
    zzzz_term *= C_zz_j[None,None,:,:]
    zzzz_term = np.einsum('jnkm->jkmn', zzzz_term)
    cov_lzzlzz_term2 *= zzzz_term[None,:,:,None,:,:]
    #cov_lzzlzz_term2 = np.einsum('ijnlkm->ijklmn',cov_ll_term2[:,None,None,:,None,None]*C_zz_i[None,:,:,None,None,None]*C_zz_j[None,None,None,None,:,:])
    return cov_lzzlzz_term2

def Tristram_approx_Nmt_separable_clzz_2components(nmt_covariance_workspace, C_l_i, C_l_j, C_zz_i, C_zz_j, nmt_workspace, nmt_binning):
    cov_lzzlzz = Tristram_approx_Nmt_separable_clzz_2components_term_acbd(nmt_covariance_workspace, C_l_i, C_l_j, C_zz_i, C_zz_j, nmt_workspace, nmt_binning)
    cov_lzzlzz += Tristram_approx_Nmt_separable_clzz_2components_term_adbc(nmt_covariance_workspace, C_l_i, C_l_j, C_zz_i, C_zz_j, nmt_workspace, nmt_binning)
    return cov_lzzlzz

def separate_separable_clzz(C_lzz):
   return [C_lzz[:,0,0], C_lzz[3,:,:]/C_lzz[3,0,0]]

def Tristram_approx_Nmt_separable_clzz_components_list(nmt_covariance_workspace, separable_cl_zz_list, nmt_workspace, nmt_binning):
    #Assuming auto-power for now. Re-write generally later.
    num_components = len(separable_cl_zz_list)
    half_shape = (nmt_workspace.wsp.bin.n_bands, separable_cl_zz_list[0].shape[1], separable_cl_zz_list[0].shape[2])
    cov_lzzlzz = np.zeros(half_shape + half_shape)
    #cov_temp = np.ones_like(cov_lzzlzz)
    for ind in range(num_components):
        C_lzz_i = separate_separable_clzz(separable_cl_zz_list[ind])
        #cov_lzzlzz += Tristram_approx_Nmt_separable_clzz_2components(nmt_covariance_workspace, C_lzz_i[0], C_lzz_i[0], C_lzz_i[1], C_lzz_i[1], nmt_workspace, nmt_binning)
        #for ind2 in range(ind+1, num_components):
        for ind2 in range(num_components):
            C_lzz_j = separate_separable_clzz(separable_cl_zz_list[ind2])
            #cov_piece = Tristram_approx_Nmt_separable_clzz_2components(nmt_covariance_workspace, C_lzz_i[0], C_lzz_j[0], C_lzz_i[1], C_lzz_j[1], nmt_workspace, nmt_binning)
            #print(cov_temp.shape)
            #cov_temp[:] = 1.
            Tristram_approx_Nmt_separable_clzz_2components_sym_update_cov(nmt_covariance_workspace, C_lzz_i[0], C_lzz_j[0], C_lzz_i[1], C_lzz_j[1], nmt_workspace, nmt_binning, cov_lzzlzz, cov_lzzlzz)
            #cov_lzzlzz += cov_piece
            #cov_lzzlzz += np.einsum('ijklmn->imnljk', cov_piece)
    return cov_lzzlzz

def get_zz_flattened_clzz_auto(clzz, flip_order=False):
    indeces = np.triu_indices(clzz.shape[1],m=clzz.shape[2])
    if not flip_order:
        return cl_zz[:,indeces[0],indeces[1]]
    else:
        return cl_zz[:,indeces[1],indeces[0]]

def get_zz_flattened_cov_auto(cov, flip_order=False):
    indeces = np.triu_indices(cov.shape[1],m=cov.shape[2])
    if not flip_order:
        return (cov[:,indeces[0],indeces[1],:,:,:])[:,:,:,indeces[0],indeces[1]]
    else:
        return (cov[:,indeces[1],indeces[0],:,:,:])[:,:,:,indeces[1],indeces[0]]


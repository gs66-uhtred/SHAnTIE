import numpy as np
import healpy as hp
from numba import njit

def get_real_ylm(l, m, nside):
    m_abs = np.abs(m)
    alms = np.zeros(  ((l+1)*(l+2) - (l-m_abs)*(l-m_abs+1))//2 , dtype = complex)
    if m>0:
        alms[-1] = 2**0.5*(-1)**m
    elif m==0:
        alms[-1] = 2**0.5
    else:
        alms[-1] = 1j*(2**0.5*(-1)**(-m))
    ylm = hp.alm2map(alms, nside=nside, pol=False, lmax=l, mmax=m_abs)
    return ylm

def get_coupling_to_lm(real_ylm, lmax, weights, m0 = False, keep_real = True):
    ans = hp.sphtfunc.map2alm(weights*real_ylm, lmax)
    if keep_real and not m0:
        ans = 2**(-0.5)*(np.real(ans) + np.imag(ans))
    elif not keep_real:
        ans = 2**(-0.5)*ans
    else:
        ans = np.real(ans)
    return ans

def compute_real_coupling_kernel(weights, nside):
    lmax = 3*nside - 1
    l_array = np.arange(lmax + 1)
    lms = hp.Alm.getlm(lmax, i=None)
    dimension = len(lms[0])
    m_coupling_kernel = np.zeros((2*dimension, 2*dimension))
    for ind in range(dimension):
        ylm_pos = get_real_ylm(lms[0][ind], lms[1][ind], nside)
        temp = get_coupling_to_lm(ylm_pos, lmax, weights, keep_real = False)
        m_coupling_kernel[:dimension,ind] = np.real(temp)
        m_coupling_kernel[dimension:,ind] = np.imag(temp)
        if lms[1][ind] != 0:
            ylm_neg = get_real_ylm(lms[0][ind], -lms[1][ind], nside)
            temp = get_coupling_to_lm(ylm_neg, lmax, weights, keep_real = False)
            m_coupling_kernel[:dimension,dimension + ind] = np.real(temp)
            m_coupling_kernel[dimension:,dimension + ind] = np.imag(temp)
    return m_coupling_kernel

@njit
def compute_cl_coupling_from_kernel(ls, kernel1, kernel2, cl):
    dl_coupling= np.zeros(kernel1.shape)
    for ind in range(kernel1.shape[0]):
        ind_modulo = ind%len(ls)
        for ind1 in range(kernel1.shape[0]):
            for ind2 in range(kernel1.shape[0]):
                dl_coupling[ind1,ind2] += cl[ls[ind_modulo]]*kernel1[ind1,ind]*kernel2[ind2,ind]
    return dl_coupling

@njit
def pcl_cov_piece_from_dl_couplings(ls, coupling1, coupling2, n_l):
    pcl_cov = np.zeros((n_l, n_l))
    for ind1, l1 in enumerate(ls):
        for ind2, l2 in enumerate(ls):
            pcl_cov[l1,l2] += coupling1[ind1,ind2]*coupling2[ind1,ind2]
    for l1 in range(n_l):
        for l2 in range(n_l):
            pcl_cov[l1,l2] /= (2*l1+1)*(2*l2+1)
    return pcl_cov

def compute_l_l_l_coupling_term(weights1, nside, weights2=None):
    print('new test')
    lmax = 3*nside - 1
    l_array = np.arange(lmax + 1)
    print(lmax)
    print(len(l_array))
    lms = hp.Alm.getlm(lmax, i=None)
    print(lms)
    coupling_term = np.zeros((lmax+1, lmax+1, lmax+1))
    for l in l_array:
        #n_ells = 2*l + 1
        for m in range(l+1):
            ylm_pos = get_real_ylm(l,m, nside)
            couple_piece1 = get_coupling_to_lm(ylm_pos, lmax, weights1, m==0)
            if weights2 != None:
                couple_piece2 = get_coupling_to_lm(ylm_pos, lmax, weights2, m==0)
            else:
                couple_piece2 = couple_piece1
            fill_coupling(coupling_term[:,:,l], couple_piece1, couple_piece2, lms[0], lms[1])
            if m != 0:
                ylm_neg = get_real_ylm(l,-m, nside)
                neg_couple1 = get_coupling_to_lm(ylm_neg, lmax, weights1)
                couple_piece1 = neg_couple1
                if weights2 != None:
                    couple_piece2 = get_coupling_to_lm(ylm_neg, lmax, weights2)
                else:
                    couple_piece2 = neg_couple1
                fill_coupling(coupling_term[:,:,l], couple_piece1, couple_piece2, lms[0], lms[1])
        #coupling_term[:,:,l]/n_ells
    n_ells = (2*l_array + 1)**0.5
    coupling_term = coupling_term/(n_ells[:,None,None]*n_ells[None,:,None])
    return coupling_term
            
@njit
def fill_coupling(coupling, piece1, piece2, ls, ms):
    l_length = np.shape(coupling)[0]
    for ind1, l1 in enumerate(ls):
        for ind2, l2 in enumerate(ls):
            #print(ind1)
            #print(ind2)
            coupling[l1,l2] += piece1[ind1]*piece2[ind2]
    #for l1 in range(l_length):
    #    n_l1 = 2*l1+1
    #    for l2 in range(l_length):
    #        n_l2 = 2*l2+1
    #        coupling[l1,l2] /= (n_l1*n_l2)

def unbinned_cov_from_l_l_l_coupling(coupling, cl):
    #Assuming auto-power for now.
    #full_coupling = coupling[:,:,:,None]*coupling[:,:,None,:]
    #coeff = np.sum(np.sum((full_coupling*cl[None,None,None,:]*cl[None,None,:,None]), axis = -1), axis = -1)
    coeff = np.sum(coupling*cl[None,None,:], axis = -1)
    return 2*coeff**2

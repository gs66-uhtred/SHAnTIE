#Module to compute Cl(z,z') from P(k,z) and redshift function phi(z) using Limber approximation.
import numpy as np

def cl00_integrand(ells, phi, pkz, ks):
    #ks is a vector of k values over which to compute this integrand.
    #pkz is a 2-dim array of the power spectrum at z = z(r=ells/ks), k=ks, shape [ells, ks]. If getting from function, plug in these values first.
    #phi(z_vect) returns matrix of all redshift window functions for all z-bins and for input redshifts z_vect, shape is [z_bin, vector_of_z_values].
    #Dimensions of answer will be [ells, ks, z, z']
    ans = (1./ells[:,None,None,None])*(phi(ells[:,None]/ks[None,:]))[:,:,:,None]*(phi(ells[:,None]/ks[None,:]))[:,:,None,:]*pkz[:,:,None,None]
    return ans

def integrate_over_k(integrand, ks, trap_rule = True, axis = 1):
    #Integrand dimensions should be [ells, ks, z, z'].
    if trap_rule:
        ans = np.trapz(integrand, x=ks, axis = axis)
    else:
        ans = None
    return ans

def tophat_phi_from_z_edges(z_edges, normalize_by_width = False):
    #z_edges must be ordered from low to high.
    #If r edges is given in place of z_edges, and normalize_by_width is used, this will return phi_r.
    def phi(z):
        #Shape of returned answer is shape of z, followed by number of z bins.
        z_diff = z_edges[1:] - z_edges[0:-1]
        n_z = len(z_diff)
        final_shape = np.ones((n_z))[...,None]*z_edges[1:]
        z_bool = np.logical_and(z[...,None]<=z_edges[1:], z[...,None]>=z_edges[0:-1]).astype(float)
        if normalize_by_width:
            z_bool = z_bool/z_diff
        return z_bool
    return phi

def find_relevant_k_ranges(r_edges, ells, rsd_edges = False):
    if not rsd_edges:
        extra_ell_factor = 0
    else:
        extra_ell_factor = 2
    #r_edges should be a 1d array of the tophat selection function edges ordered from low to high.
    r_mins = r_edges[:-1]
    r_maxs = r_edges[1:]
    k_mins = (ells[None,:]+0.5-extra_ell_factor)/r_maxs[:,None]
    k_maxs = (ells[None,:]+0.5+extra_ell_factor)/r_mins[:,None]
    return k_mins, k_maxs

def find_relevant_ks(r_edges, ells, rsd_edges = False, num_ks = 100, spacing = 'linear'):
    #r_edges should be a 1d array of the tophat selection function edges ordered from low to high.
    #r_mins = r_edges[:-1]
    #r_maxs = r_edges[1:]
    #k_mins = ells[None,:]/r_maxs[:,None]
    #k_maxs = ells[None,:]/r_mins[:,None]
    k_mins, k_maxs = find_relevant_k_ranges(r_edges, ells, rsd_edges = rsd_edges)
    if spacing == 'logarithmic':
        ks = np.logspace(np.log10(k_mins), np.log10(k_maxs), num=num_ks)
    else:
        ks = np.linspace(k_mins, k_maxs, num=num_ks)
    return ks

def F0(ells, ks, phi):
    #ells is an ndarray of ell values, shape [num_ks_per_z_channel, num_z_channels, num_ells]
    #ks is an ndarray of k values, shape [num_ks_per_z_channel, num_z_channels, num_ells]
    #phi is a function that takes in an array of rs and returns an array of shape [shape of rs, num_z_channels]
    prefactor = (2*ells**2 + 2*ells - 1)/((2*ells-1)*(2*ells+3)*(ells+0.5)**0.5)
    return prefactor*np.einsum('ijkj->ijk', phi((ells+0.5)/ks))

def Fn2(ells, ks, phi):
    #ells is an ndarray of ell values, shape [num_ks_per_z_channel, num_z_channels, num_ells]
    #ks is an ndarray of k values, shape [num_ks_per_z_channel, num_z_channels, num_ells]
    #phi is a function that takes in an array of rs and returns an array of shape [shape of rs, num_z_channels]
    prefactor = -(ells*(ells - 1))/((2*ells-1)*(2*ells+1)*(ells+0.5-2)**0.5)
    return prefactor*np.einsum('ijkj->ijk', phi((ells+0.5-2)/ks))

def Fp2(ells, ks, phi):
    #ells is an ndarray of ell values, shape [num_ks_per_z_channel, num_z_channels, num_ells]
    #ks is an ndarray of k values, shape [num_ks_per_z_channel, num_z_channels, num_ells]
    #phi is a function that takes in an array of rs and returns an array of shape [shape of rs, num_z_channels]
    prefactor = -((ells+1)*(ells + 2))/((2*ells+1)*(2*ells+3)*(ells+0.5+2)**0.5)
    return prefactor*np.einsum('ijkj->ijk', phi((ells+0.5+2)/ks))

def F_total(ells, ks, phi):
    return F0(ells, ks, phi) + Fn2(ells, ks, phi) + Fp2(ells, ks, phi)

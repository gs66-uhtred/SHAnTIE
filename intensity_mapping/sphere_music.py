import scipy as sp
import numpy as np
import healpy as hp
from scipy import special
from scipy import linalg as sl
# pylint: disable=no-member, no-name-in-module
"""
TODO:
add unit test: mixing identity when uniform mask
add lmax to mixing matrix calculation
try anafast with regression option to keep monopole
can crosspower mixing be done with the same function?
"""


def wigner3j_red1(j, m):
    """ Evaluate:
    j  j 1
    m -m 0

    See Edmonds p. 125
    >>> np.abs(wigner3j_red1(1, 0))
    0.0
    >>> wigner3j_red1(1, 1)
    -0.408248290...
    >>> wigner3j_red1(350, 3)
    -0.00032327689...
    >>> wigner3j_red1(1000, 3)
    -6.70317675924...e-05
    """
    return -1. ** (j - m) * m / np.sqrt((2. * j + 1.) * (j + 1.) * j)


def wigner3j_mzero(j1, j2, j3):
    """Evaluate:
    j1 j2 j3
    0  0  0

    scipy with gammaln
    Direct "analytic" method: See p. 50 in Edmonds:
    "Angular momentum in quantum mechanics"
    this seems to be the fastest algorithm supporting high j
    for low j, standard math is fine and faster

    >>> wigner3j_mzero(sp.array([100,200,300]), \
                       sp.array([100,200,300]), \
                       sp.array([200,300,400]))
    array([ 0.01409259,  0.00282467,  0.00188342])
    """
    j = j1 + j2 + j3
    g = j / 2

    ret = special.gammaln(2 * (g - j1) + 1)
    ret += special.gammaln(2 * (g - j2) + 1)
    ret += special.gammaln(2 * (g - j3) + 1)
    ret -= special.gammaln(j + 2)
    ret *= 0.5
    ret += special.gammaln(g + 1)
    ret -= special.gammaln(g - j1 + 1)
    ret -= special.gammaln(g - j2 + 1)
    ret -= special.gammaln(g - j3 + 1)
    ret = sp.exp(ret)

    sign = sp.ones_like(j)
    sign[sp.mod(g, 2) == 1] = -1
    ret[sp.mod(j, 2) != 0] = 0.
    return ret * sign


def wigner3j(j1, j2, j3, m1, m2, m3):
    """Evaluate full Wigner 3J:
    j1 j2 j3
    m1 m2 m3

    See: http://dlmf.nist.gov/34.2
    this uses gammaln for factorials and conditions the exponential sum
    This is not shown in the NIST equations

    This fails for large arguments; see comparison with wigner_mzero

    TODO:
    This gives the opposite sign from wigner3j_red1(1, 1)
    check j1+j2+j3 in integers/parity?
    add keywords to force checks

    repeat wigner3j_red1
    >>> wigner3j(1, 1, 1, 0, 0, 0)
    0.0
    >>> wigner3j(1, 1, 1, 1, -1, 0)
    0.408248290...
    >>> wigner3j(350, 350, 1, 3, -3, 0)
    -0.00032327...
    >>> wigner3j(1000, 1000, 1, 3, -3, 0)
    -6.70317675...e-05

    repeat wigner_mzero
    >>> wigner3j(100, 100, 200, 0, 0, 0)
    0.01409258...
    >>> wigner3j(200, 200, 300, 0, 0, 0)
    1.41173827...e+19
    """
    if ((2 * j1 != np.floor(2 * j1)) or \
        (2 * j2 != np.floor(2 * j2)) or \
        (2 * j3 != np.floor(2 * j3)) or \
        (2 * m1 != np.floor(2 * m1)) or \
        (2 * m2 != np.floor(2 * m2)) or \
        (2 * m3 != np.floor(2 * m3))):
        print('All arguments must be integers or half-integers.')
        return None

    if ((m1 + m2 + m3) != 0):
        print('3j-Symbol unphysical')
        return 0

    if ((j1 - m1) != np.floor(j1 - m1)):
        print('2*j1 and 2*m1 must have the same parity')
        return 0

    if ((j2 - m2) != np.floor(j2 - m2)):
        print('2*j2 and 2*m2 must have the same parity')
        return; 0

    if ((j3 - m3) != np.floor(j3 - m3)):
        print('2*j3 and 2*m3 must have the same parity')
        return 0

    if (j3 > j1 + j2) or (j3 < np.abs(j1 - j2)):
        print('j3 is out of bounds.')
        return 0

    if np.abs(m1) > j1:
        print('m1 is out of bounds.')
        return 0

    if np.abs(m2) > j2:
        print('m2 is out of bounds.')
        return 0

    if np.abs(m3) > j3:
        print('m3 is out of bounds.')
        return 0

    t1 = j2 - m1 - j3
    t2 = j1 + m2 - j3
    t3 = j1 + j2 - j3
    t4 = j1 - m1
    t5 = j2 + m2

    tmin = max(0, max(t1, t2))
    tmax = min(t3, min(t4, t5))
    s_indices = np.arange(tmin, tmax + 1, 1)

    # condition the exponential sum by dividing by its largest term
    # subtract the log of that from the log of the main term below
    dlist = []
    for s in s_indices:
        denom = sp.special.gammaln(s + 1)
        denom += sp.special.gammaln(t3 - s + 1)
        denom += sp.special.gammaln(t4 - s + 1)
        denom += sp.special.gammaln(t5 - s + 1)
        denom += sp.special.gammaln(s - t1 + 1)
        denom += sp.special.gammaln(s - t2 + 1)
        dlist.append(denom)

    cond = np.min(dlist)
    rsum = 0
    for s, denom in zip(s_indices, dlist):
        rsum += (-1.)**s * np.exp(-(denom - cond))

    term = sp.special.gammaln(j1 + j2 - j3 + 1)
    term += sp.special.gammaln(j1 - j2 + j3 + 1)
    term += sp.special.gammaln(-j1 + j2 + j3 + 1)
    term -= sp.special.gammaln(j1 + j2 + j3 + 2)

    term += sp.special.gammaln(j1 + m1 + 1)
    term += sp.special.gammaln(j1 - m1 + 1)
    term += sp.special.gammaln(j2 + m2 + 1)
    term += sp.special.gammaln(j2 - m2 + 1)
    term += sp.special.gammaln(j3 + m3 + 1)
    term += sp.special.gammaln(j3 - m3 + 1)
    term = np.exp(0.5 * term - cond)

    return rsum * (-1.) ** (j1 - j2 - m3) * term


def mixing_matrix(ell, weight_wl, progress=False):
    """ell-ell' mixing matrix
    This implements A31 in 2002ApJ...567....2H

    >>> ell = np.array([0, 1, 2])
    >>> wl = np.array([0.1, 0.2, 0.3])
    >>> print mixing_matrix(ell, wl)
    [[ 0.00795775  0.01591549  0.02387324]
     [ 0.04774648  0.05570423  0.01909859]
     [ 0.11936621  0.03183099  0.04206238]]
    """
    jgrid = np.meshgrid(ell, ell)
    jgrid_row = np.copy(jgrid[1])
    jgrid_col = np.copy(jgrid[0])
    jgrid[0] = jgrid_row
    jgrid[1] = jgrid_col
    mixing = np.zeros_like(jgrid[0], dtype=float)

    # sum on l_3 "fixed_ell"
    for index, fixed_ell in enumerate(ell):
        if progress:
            print(index, end=' ')

        # evaluate Wigner 3J at the matrix of all l1 l2
        term = wigner3j_mzero(jgrid[0], jgrid[1], fixed_ell) ** 2.
        # set any infinite terms to 0.
        term[np.invert(np.isfinite(term))] = 0.
        #term[np.isnan(term)] = 0.

        # multiply by the normalization and weight
        term *= (2. * fixed_ell + 1)
        term *= weight_wl[index]

        mixing += term

    mixing *= (2. * jgrid[1] + 1) / (4. * np.pi)
    return mixing

def weight_coupling_kernel(weight, lmax = None, m2_negative = False, n_processes = 1, this_process = 0):
    # If m2_negative is true, calculate coupling from negative m2 to positive m1.
    # If m2_negative is true, also zero out coupling from m2=0 to avoid double counting.
    from intensity_mapping import real_alm as ra 
    import py3nj
    wlms = hp.sphtfunc.map2alm(weight, lmax=lmax)
    if lmax == None:
        lmax = 3*(np.size(weight)/12)**0.5 - 1
    #print lmax
    ilen = ra.ilength(lmax,lmax)
    lm_index = np.arange(ilen)
    (lvec, mvec) = ra.i2lm(lmax, lm_index)
    #print mvec.shape
    #print lvec.shape
    coup_kernel = np.zeros((np.size(wlms), np.size(wlms)), dtype = np.complex)
    #print coup_kernel.shape
    lvec = lvec.astype(int)
    #print lvec[0:10]
    #print mvec[0:10]
    #print wlms.shape
    #print np.max(lvec)
    #print np.max(mvec)
    lvec_col = (lvec[None,:]*np.ones((lvec.size,lvec.size), dtype = 'int')).reshape((lvec.size**2))
    lvec_row = (lvec[:,None]*np.ones((lvec.size,lvec.size), dtype = 'int')).reshape((lvec.size**2))
    mvec_col = (mvec[None,:]*np.ones((lvec.size,lvec.size), dtype = 'int')).reshape((lvec.size**2))
    mvec_row = (mvec[:,None]*np.ones((lvec.size,lvec.size), dtype = 'int')).reshape((lvec.size**2))
    if m2_negative:
        mvec_col = -mvec_col
    split_wlms = np.array_split(wlms, n_processes)[this_process]
    split_indeces = np.array_split(np.arange(wlms.size), n_processes)[this_process]
    #print split_wlms.shape
    #print split_indeces.shape
    for index, wlm in enumerate(split_wlms):
        print(split_indeces[index])
        factor1 = wlm*(-1)**mvec[None,:]*((2*lvec[:,None]+1)*(2*lvec[None,:]+1)*(2*lvec[split_indeces[index]][None, None]+1)/(4*np.pi))**0.5
        factor2 = py3nj.wigner3j(2*lvec_row, 2*lvec_col, 2*lvec[split_indeces[index]], 0, 0, 0).reshape((lvec.size,lvec.size))
        factor3 = py3nj.wigner3j(2*lvec_row, 2*lvec_col, 2*lvec[split_indeces[index]], 2*mvec_row, -2*mvec_col, 2*mvec[split_indeces[index]]).reshape((lvec.size,lvec.size))
        coup_kernel += factor1*factor2*factor3
        #If m3!=0, add contribution from -m3.
        if mvec[split_indeces[index]] != 0:
            factor1 = np.conj(factor1)
            factor3 = py3nj.wigner3j(2*lvec_row, 2*lvec_col, 2*lvec[split_indeces[index]], 2*mvec_row, -2*mvec_col, -2*mvec[split_indeces[index]]).reshape((lvec.size,lvec.size))
            coup_kernel += factor1*factor2*factor3
    if m2_negative:
        coup_kernel[mvec_col.reshape((lvec.size,lvec.size)) == 0] = 0
    return coup_kernel

def wigner_package_test(l3, l3_prime, m3, m3_prime):
    from intensity_mapping import real_alm as ra
    import py3nj
    lmax = max(l3, l3_prime)
    ilen = ra.ilength(lmax,lmax)
    lm_index = np.arange(ilen)
    lvec, mvec = ra.i2lm(lmax, lm_index)
    lvec = lvec.astype(int)
    #Extend to include negative m
    lvec_extend = lvec[mvec!=0]
    mvec_extend = -mvec[mvec!=0]
    lvec = np.concatenate((lvec, lvec_extend))
    mvec = np.concatenate((mvec, mvec_extend))
    print(lvec)
    print(mvec)
    ans = np.zeros((lmax+1, lmax+1))
    lvec_col = (lvec[None,:]*np.ones((lvec.size,lvec.size), dtype = 'int')).reshape((lvec.size**2))
    lvec_row = (lvec[:,None]*np.ones((lvec.size,lvec.size), dtype = 'int')).reshape((lvec.size**2))
    mvec_col = (mvec[None,:]*np.ones((lvec.size,lvec.size), dtype = 'int')).reshape((lvec.size**2))
    mvec_row = (mvec[:,None]*np.ones((lvec.size,lvec.size), dtype = 'int')).reshape((lvec.size**2))    
    factor1 = py3nj.wigner3j(2*lvec_row,2*lvec_col, 2*l3, 2*mvec_row, 2*mvec_col, 2*m3)
    factor2 = py3nj.wigner3j(2*lvec_row,2*lvec_col, 2*l3_prime, 2*mvec_row, 2*mvec_col, 2*m3_prime)
    for l1 in np.arange(lmax+1):
        for l2 in np.arange(lmax+1):
            l1_l2_bool = np.logical_and(lvec_row == l1, lvec_col == l2)
            intermediate = (factor1*factor2)[l1_l2_bool]
            print('For l1=' + str(l1) + ' and l2=' + str(l2) + ', there are ' + str(np.size(intermediate)) + ' m1,m2 pairs.')
            print('m1 is ' + str(mvec_row[l1_l2_bool]))
            print('m2 is ' + str(mvec_col[l1_l2_bool]))
            print('Wigner 3j for l3 and m3 is ' + str(factor1[l1_l2_bool]))
            print('Wigner 3j for l3_prime and m3_prime is ' + str(factor2[l1_l2_bool]))
            ans[l1,l2] = np.sum(intermediate)
    return ans


def quad_coupling_kernel(ell, w_ab, w_cd, progress=False):
    """ This implements equation 27 in astro-ph/0405575 """
    jgrid = np.meshgrid(ell, ell)
    jgrid_row = np.copy(jgrid[1])
    jgrid_col = np.copy(jgrid[0])
    jgrid[0] = jgrid_row
    jgrid[1] = jgrid_col
    #mixing = np.zeros_like(jgrid[0], dtype=float)
    mixing = np.zeros_like(jgrid[0], dtype=np.complex128)

    max_l = np.max(ell)
    l, m = hp.sphtfunc.Alm.getlm(max_l)
    # sum on l_3 "fixed_ell"
    for index, fixed_ell in enumerate(ell):
        if progress:
            print(index, end=' ')

        # evaluate Wigner 3J at the matrix of all l1 l2
        term = wigner3j_mzero(jgrid[0], jgrid[1], fixed_ell) ** 2.
        # set any infinite terms to 0.
        term[np.invert(np.isfinite(term))] = 0.
        #term[np.isnan(term)] = 0.

        term=term.astype(np.complex128)

        # multiply by the normalization and weight
        l_and_mzero_bool = np.logical_and(l == fixed_ell, m == 0)
        l_and_mnonzero_bool = np.logical_and(l == fixed_ell, m != 0)
        term *= np.sum((w_ab*np.conj(w_cd))[l_and_mzero_bool]) + 2.*np.sum((w_ab*np.conj(w_cd))[l_and_mnonzero_bool])
        #term *= (2. * fixed_ell + 1)

        mixing += term

    mixing *= (2. * jgrid[1] + 1) / (4. * np.pi)
    return mixing


def master_auto_test_prep(weightfile, mixingfile):
    """calculate the mixing matrix based on files"""
    mask_map = hp.read_map(weightfile)
    mixing_matrix = mixing_from_mask(mask_map)
    np.save(mixingfile, mixing_matrix)


def mixing_from_weight(weight_left, weight_right, lmax=None, iter=3):
    mask_cls = hp.sphtfunc.anafast(weight_left, map2=weight_right, lmax=lmax, iter=iter)
    ell = np.arange(len(mask_cls))
    n_pix = weight_left.shape[0]

    # put the monopole back in
    #monopole = np.sum(weight_left) / float(n_pix)
    #monopole *= np.sum(weight_right) / float(n_pix)
    #mask_cls[0] = monopole * (4. * np.pi)

    return mixing_matrix(ell, mask_cls)


def eigh_inverse(cov, cond_limit=None):
    """Calculate the inverse of a hermitian (e.g. symmetric covariance)
    matrix using the eigenvector decomposition. Also return the log det.
    """
    values, eig_vecs = sl.eigh(cov)
    nonzero_vals = np.copy(values)
    wh_bad = (values <= 0.)
    nonzero_vals[wh_bad] = 1.
    inv_values = 1. / nonzero_vals
    inv_values[wh_bad] = 0.

    if cond_limit is not None:
        # eigenvalues are sorted in ascending order
        cond_vec = values / np.max(values)
        inv_values[cond_vec < cond_limit] = 0.
        # truncate rather than null so log works
        values = values[cond_vec > cond_limit]
        #print "ncut: ", np.sum(cond_vec < cond), np.min(cond_vec)
    else:
        # still condition the eigenvalues to avoid ln0 and 1/0.
        values = values[values > 0.]
        nbad = np.sum(values <= 0.)
        if nbad > 0:
            logging.warning("There are %d 0 eigenvectors", nbad)
        #inv_values[values > 0.] = 0.

    invcov = np.dot(eig_vecs, (inv_values * eig_vecs).T)
    lndet = np.sum(np.log(values))

    return lndet, invcov


if __name__ == "__main__":
    import doctest

    OPTIONFLAGS = (doctest.ELLIPSIS |
                   doctest.NORMALIZE_WHITESPACE)
    doctest.testmod(optionflags=OPTIONFLAGS)

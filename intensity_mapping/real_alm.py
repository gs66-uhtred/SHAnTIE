"""HealPy alm arrays are ordered as follows.
In this case, ellmax = 5.

             m_ell value ->
             0   1   2   3   4   5

ell    0     0
value  1     1   6
|      2     2   7  11
|      3     3   8  12  15
V      4     4   9  13  16  18
       5     5  10  14  17  19  20

Negative m_ell values are not used; because the real-valued nature of the
temperature field constrains the negative m_ell valued coefficients once the
positive-valued coefficients are known.

The desired ordering of real-valued coefficients is

                             m_ell value ->
            -4  -3  -2  -1   0   1   2   3   4

ell    0                     0
value  1                 1   2   3
|      2             4   5   6   7   8
|      3         9  10  11  12  13   14  15
V      4    16  17  18  19  20  21   22  23  24

I could do something to try to mimic the HealPy ordering instead, but if I have
to rearrange the ordering to make them real-valued anyway, I might as well pick
an ordering that I'm used to (and that is easily extendable to higher ell,
without reordering the array).  Most of the time, I won't be working with
m_ell-truncated instrument beams, which is what the HealPy ordering is very good
for.

We can use the same ordering for the real-valued harmonics as for the polarized
E and B modes, because (except for the monopole and dipole) there is a natural
correspondence between the two.

Note that we use "m_ell" instead of the "m" to conform with PEP8 standards.
"""

import numpy as np
import healpy as hp
import unittest




def lm2i(ellmax, ell, m_ell):
    """
    Conversion from multipole ell and m_ell (where m_ell >= 0) to the index i
    used in healpy ordering of alm values.
    """
    i = m_ell * (ellmax + 1) - ((m_ell - 1) * m_ell) / 2 + ell - m_ell
    return i


def i2lm(ellmax, i):
    """
    Conversion from index i used for healpy ordering of alm values
    to multipole ell and m_ell.
    """
    m_ell = (np.floor(ellmax + 0.5 - np.sqrt((ellmax + 0.5)**2 - \
            2 * (i - ellmax - 1)))).astype(int) + 1
    ell = i - m_ell * (ellmax + 1) + ((m_ell - 1) * m_ell) / 2 + m_ell
    return (ell, m_ell)


def ilength(ellmax, mmax):
    """
    The number of elements in a healpy alm array with a given ellmax and mmax
    """
    assert ellmax >= mmax
    assert ellmax >= 0
    assert mmax >= 0
    return int((mmax + 1) * (ellmax + 1) - ((mmax + 1) * mmax) / 2)


# Let's use the index r_lm as the real-valued ordering integer.
# Note that we don't need to know ellmax.
def r2lm(r_lm):
    """
    Conversion from real-valued ordering index r_lm to multipole ell and m_ell.
    -ell <= m_ell <= ell
    ell >= 0
    """
    ell = np.floor(np.sqrt(r_lm)).astype(int)
    m_ell = r_lm - ell**2 - ell
    return (ell, m_ell)

# m_ell is allowed to be negative, here.
def lm2r(ell, m_ell):
    """
    Conversion from multipole ell and m_ell to real-valued ordering index r.
    """
    r_lm = ell**2 + ell + m_ell
    assert np.floor(r_lm) == r_lm
    return int(r_lm)


# I'm sure this can be optimized for speed, later.
# At low ell, we don't really care; it will be fast enough.
def complex2real(ellmax, mmax, cx_alm):
    """
    Convert an array from HealPy of complex-valued alms into an array
    of real-valued alms.
    """
    length = ilength(ellmax, mmax)
    assert cx_alm.shape[0] == length
    #print "len = ", length
    re_alm = np.zeros((ellmax+1)**2, dtype='double')
    for i in range(length):
        ell, m_ell = i2lm(ellmax, i)
        r_lm = lm2r(ell, m_ell)
        if m_ell == 0:
            re_alm[r_lm] = np.real(cx_alm[i])
        else:
            re_alm[r_lm] = np.real(cx_alm[i]) * np.sqrt(2.0)
            r_lm = lm2r(ell, -m_ell)
            re_alm[r_lm] = (-1)**m_ell * np.imag(cx_alm[i]) * np.sqrt(2.0)
    return re_alm


def real2complex(re_alm, ellmax=-1, mmax=-1):
    """
    Convert an array of real-valued alms to a healpy array of complex-valued
    alms.
    """
    if ellmax < 0:
        ellmax = np.floor(np.sqrt(re_alm.shape[0] - 1)).astype(int)
    if mmax < 0:
        mmax = ellmax
    assert re_alm.shape[0] >= (ellmax+1)**2
    length = ilength(ellmax, mmax)
    cx_alm = np.zeros(length, dtype='cfloat')
    for i in range(length):
        ell, m_ell = i2lm(ellmax, i)
        r1_lm = lm2r(ell, m_ell)
        if m_ell == 0:
            cx_alm[i] = re_alm[r1_lm]
        else:
            r2_lm = lm2r(ell, -m_ell)
            cx_alm[i] = (re_alm[r1_lm] + (-1)**m_ell * 1.0j * re_alm[r2_lm]) \
                    / np.sqrt(2.0)
    return cx_alm


def real2cl(re_alm):
    """
    Return raw cl values, from an array of real-valued alms.
    """
    cx_alm = real2complex(re_alm)
    c_ell = hp.sphtfunc.alm2cl(cx_alm)
    return c_ell



class TestIndexing(unittest.TestCase):
    """
    class for testing that indexing works correctly
    """
    def my_test_i(self, ellmax, i):
        """
        Tests that i2lm() and lm2i() are inverse of each other
        """
        ell, m_ell = i2lm(ellmax, i)
        self.assertTrue(m_ell >= 0)
        self.assertTrue(ell >= 0)
        self.assertTrue(m_ell <= ell)
        self.assertTrue(ell <= ellmax)
        i_2 = lm2i(ellmax, ell, m_ell)
        self.assertTrue(i == i_2)

    def my_test_r(self, r_lm):
        """
        Checks that r2lm() and lm2r() are inverses of each other
        """
        ell, m_ell = r2lm(r_lm)
        self.assertTrue(ell >= 0)
        self.assertTrue(abs(m_ell) <= ell)
        r2_lm = lm2r(ell, m_ell)
        self.assertTrue(r_lm == r2_lm)

    def my_test_lm(self, ellmax, ell, m_ell):
        """
        checks that ell and m_ell are never greater than ell_max, and that
        i2lm() and lm2i() give compatible results
        """
        self.assertTrue(m_ell <= ell)
        i = lm2i(ellmax, ell, m_ell)
        ell_2, m_2 = i2lm(ellmax, i)
        #print "in:  ", i, ell, m
        #print "out: ", i, ell2, m2
        self.assertTrue(i >= 0)
        self.assertTrue(i < ilength(ellmax, ellmax))
        self.assertTrue(ell_2 == ell)
        self.assertTrue(m_2 == m_ell)

    def my_test_rlm(self, ell, m_ell):
        """
        Tests that lm2r() gives consistent result with r2lm()
        """
        self.assertTrue(abs(m_ell) <= ell)
        r_lm = lm2r(ell, m_ell)
        ell2, m_2 = r2lm(r_lm)
        self.assertTrue(r_lm >= 0)
        self.assertTrue(r_lm < (ell + 1)**2)
        self.assertTrue(ell2 == ell)
        self.assertTrue(m_2 == m_ell)

    def test_manual(self):
        """
        Unit tests for showing there are the correct number of indices for a
        given ell.
        """
        self.assertEqual(ilength(5, 3), 18)
        self.assertEqual(ilength(5, 5), 21)
        self.assertEqual(ilength(0, 0), 1)
        self.assertEqual(ilength(1, 0), 2)
        self.assertEqual(ilength(1, 1), 3)

        self.assertEqual(lm2i(0, 0, 0), 0)
        self.assertEqual(lm2i(1, 0, 0), 0)
        self.assertEqual(lm2i(2, 0, 0), 0)

    def test_ilength(self):
        """
        Another unit test for length
        """
        self.assertEqual(ilength(95, 95), 4656)

    def test_i_lowell(self):
        """
        Runs my_test_i() for many ells.
        """
        for ellmax in np.arange(0, 10):
            for i in range(ilength(ellmax, ellmax)):
                self.my_test_i(ellmax, i)

    def test_r_lowell(self):
        """
        Runs my_test_r() for many ells.
        """
        ellmax = 10
        for r_lm in np.arange(ellmax**2):
            self.my_test_r(r_lm)

    def test_i_highell(self):
        """
        Runs my_test_i() for many ells.
        """
        ellmax = 10000
        for i in np.arange(ilength(ellmax, ellmax) - 100, \
                ilength(ellmax, ellmax)):
            self.my_test_i(ellmax, i)

    def test_r_highell(self):
        """
        Runs my_test_r() for many ells.
        """
        ellmax = 10000
        for r_lm in np.arange(ellmax**2 - 100, ellmax**2):
            self.my_test_r(r_lm)

    def test_lm_lowell(self):
        """
        Runs my_test_lm() for many ells
        """
        for ellmax in np.arange(0, 10):
            for ell in np.arange(ellmax + 1):
                for m_ell in np.arange(ell + 1):
                    self.my_test_lm(ellmax, ell, m_ell)

    def test_rlm_lowell(self):
        """
        Runs my_test_rlm() for many ells
        """
        ellmax = 10
        for ell in np.arange(ellmax + 1):
            for m_ell in np.arange(-ell, ell + 1):
                self.my_test_rlm(ell, m_ell)

    def test_lm_highell(self):
        """
        Runs my_test_lm() for many ells
        """
        ellmax = 10000
        for ell in np.arange(4):
            for m_ell in np.arange(ell + 1):
                self.my_test_lm(ellmax, ell, m_ell)
        for ell in np.arange(ellmax - 1, ellmax + 1):
            for m_ell in np.arange(ell + 1):
                self.my_test_lm(ellmax, ell, m_ell)

    def test_rlm_highell(self):
        """
        Runs my_test_rlm() for many ells
        """
        ellmax = 10000
        for ell in np.arange(ellmax - 1, ellmax + 1):
            for m_ell in np.arange(-ell, ell + 1):
                self.my_test_rlm(ell, m_ell)

    def test_array1(self):
        """
        checks for numerical errors from switching real to complex alm.
        """
        nside = 32
        npix = hp.nside2npix(nside)
        test_map = np.random.standard_normal(npix)
        c_ell, cx_alm = hp.anafast(test_map, alm=True)
        ellmax = c_ell.shape[0] - 1
        mmax = ellmax
        re_alm = complex2real(ellmax, mmax, cx_alm)
        cx_alm2 = real2complex(re_alm)
        self.assertTrue(np.max(np.abs(cx_alm2 - cx_alm)) < 1e-15)

    def test_array2(self):
        """
        checking that real and complex values have the same number of indices
        """
        ellmax = 100
        mmax = ellmax
        length = (ellmax + 1)**2
        re_alm = np.arange(length) + 1
        self.assertTrue(re_alm[0] != re_alm[1])
        self.assertTrue(re_alm[1] != re_alm[2])
        cx_alm = real2complex(re_alm)
        self.assertTrue(cx_alm.shape[0] == ilength(ellmax, mmax))
        re_alm2 = complex2real(ellmax, mmax, cx_alm)
        #print np.max(np.abs(re_alm2 - re_alm) / re_alm)
        self.assertTrue(np.min(re_alm2) > 0.5)
        self.assertTrue(np.max(np.abs(re_alm2 - re_alm) / re_alm) < 1e-14)

    def test_norm1(self):
        """
        Complex number test for large nside
        """
        nside = 64
        npix = hp.nside2npix(nside)
        ellmax = 10
        length = (ellmax + 1)**2

        for i in range(length):
            re_alm = np.zeros(length, dtype='double')
            re_alm[i] = 1.0
            cx_alm = real2complex(re_alm)
            test_map = hp.alm2map(cx_alm, nside, verbose=False)
            unity = np.sum(test_map**2) * np.pi * 4 / npix
            self.assertTrue(abs(unity - 1) < 1e-3)

    def test_orthonormal(self):
        """
        checks that real spherical harmonics are orthonormal
        """
        nside = 64
        npix = hp.nside2npix(nside)
        #ellmax = 10
        ellmax = 5
        length = (ellmax + 1)**2
        for i in range(length):
            re_alm = np.zeros(length, dtype='double')
            re_alm[i] = 1.0
            cx_alm = real2complex(re_alm)
            test_map = hp.alm2map(cx_alm, nside, verbose=False)
            unity = np.sum(test_map**2) * np.pi * 4 / npix
            self.assertTrue(abs(unity - 1) < 1e-3)
            for k in range(i+1, length):
                re_alm = np.zeros(length, dtype='double')
                re_alm[k] = 1.0
                cx_alm = real2complex(re_alm)
                test_map2 = hp.alm2map(cx_alm, nside, verbose=False)
                dot = np.sum(test_map * test_map2) * np.pi * 4 / npix
                if abs(dot) > 1e-3:
                    print((i, k, dot))
                self.assertTrue(abs(dot) < 1e-3)

    def test_cl(self):
        """
        Makes sure cl have correct behavior given alm
        """
        ellmax = 10
        length = (ellmax + 1)**2
        alm = np.zeros(length) + 2.0
        c_ell = real2cl(alm)
        for i in range(ellmax + 1):
            self.assertTrue(abs(c_ell[i] - 4.0) < 1e-14)

        alm = np.array([1, 2, 2, 2, 3, 3, 3, 3, 3])
        c_ell = real2cl(alm)
        self.assertTrue(abs(c_ell[0] - 1) < 1e-14)
        self.assertTrue(abs(c_ell[1] - 4) < 1e-14)
        self.assertTrue(abs(c_ell[2] - 9) < 1e-14)


def foo1():
    """
    Scratch paper
    """
    a_lm = hp.sphtfunc.Alm()
    ellmax = 5
    ell = 0
    m_ell = 0
    i = a_lm.getidx(ellmax, ell, m_ell)
    print(i)
    #ell, m_ell = a_lm.getlm(0, ellmax)
    #print ell, m


def foo2():
    """
    Scratch paper
    """
    nside = 32
    npix = hp.nside2npix(nside)
    test_map = np.random.standard_normal(npix)
    #c_ell, cx_alm = hp.anafast(test_map, alm=True, use_weights=True)
    c_ell, cx_alm = hp.anafast(test_map, alm=True)
    print((c_ell.shape))
    ellmax = c_ell.shape[0] - 1
    mmax = ellmax
    print((cx_alm.shape))
    print(cx_alm)
    re_alm = complex2real(ellmax, mmax, cx_alm)
    cx_alm2 = real2complex(re_alm)
    #print re_alm.shape
    #print cx_alm2.shape
    #print cx_alm2
    print((np.max(np.abs(cx_alm2 - cx_alm))))
    #print (cx_alm - cx_alm2)[-20:]

def foo3():
    """
    Scratch paper
    """
    nside = 64
    npix = hp.nside2npix(nside)
    ellmax = 10
    #mmax = ellmax
    length = (ellmax + 1)**2

    for i in range(length):
        re_alm = np.zeros(length, dtype='double')
        re_alm[i] = 1.0
        cx_alm = real2complex(re_alm)
        test_map = hp.alm2map(cx_alm, nside, verbose=False)
        #hp.mollview(test_map)
        #print np.ptp(test_map)
        #print test_map[0] - 1 / np.sqrt(np.pi * 4)
        print(("1 = ", np.sum(test_map**2) * np.pi * 4 / npix))

if __name__ == "__main__":
    foo1()
    foo2()
    unittest.main()
    foo3()


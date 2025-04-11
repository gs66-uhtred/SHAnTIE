#cimport numpy as npc
import numpy as np
import healpy as hp
from libc.math cimport fabs
from libc.math cimport cos

#def boxcar_convolve(double[:] in_map, double fwhm, double phi_tol):
def boxcar_convolve(in_map, double fwhm, double phi_tol):
    cdef int map_size = in_map.size
    cdef double[:] out_map = np.zeros(in_map.shape)
    #print 'check1'
    lon_lat = hp.pix2ang(int((map_size/12)**0.5), np.arange(in_map.size), lonlat = True)
    #print 'check2'
    cdef double[:] lon = lon_lat[0]
    cdef double[:] lat = lon_lat[1]
    cdef double lon_diff
    cdef double lat_diff
    cdef int i,j, num
    cdef double lon_diff_max = fwhm*phi_tol/2.
    cdef double lat_diff_max = fwhm/2.
    cdef double geom_fact

    for i in xrange(map_size):
        #lon_diff = fabs((lon - lon[i])%360)
        #lat_diff = fabs(lat - lat[i])
        num = 0
        geom_fact = cos(lat[i]*3.14152/180.)
        for j in xrange(map_size):
            lon_diff = fabs((lon[i] - lon[j]))
            lon_diff = min(lon_diff, 360 - lon_diff)*geom_fact
            lat_diff = fabs((lat[i] - lat[j]))
            if lon_diff <= lon_diff_max and lat_diff <= lat_diff_max:
                out_map[i] += in_map[j]
                num += 1
            lat_i_diff_90 = fabs(fabs(lat[i]) - 90)
            if lat_i_diff_90 <= lat_diff_max and lon_diff - 180*geom_fact <= lon_diff_max and lat_diff <= lat_diff_max and lat_i_diff_90 + fabs(fabs(lat[j]) - 90) <= lat_diff_max:
                out_map[i] += in_map[j]
                num += 1
        out_map[i] /= num
    return out_map
            


import numpy as np
from . import pwrspec_estimation as pe
from . import sphere_music as sm
from scipy import linalg as sl
import h5py
import glob

# rest-frame lines of interest
nu0_cii = 1897.


def co_freq(j_quantum_end):
    """simple form for the CO J lines"""
    return 115. * (j_quantum_end + 1)


def make_ang_weight(filename):
    """
    make a separable angular weight in common to all slices
    this avoids the need to recalulate the mixing matrix each time
    """
    weight_file = h5py.File(filename, "r")
    wcube = weight_file["datacube"].value
    ang_weight = np.mean(wcube, axis=1)
    ang_weight /= np.max(ang_weight)

    weight_file.close()

    return ang_weight


def make_delta(gal_fname, nbar_fname):
    """
    Given a galaxy survey count volume and nbar, find delta
    """
    gal_file = h5py.File(gal_fname, "r")
    nbar_file = h5py.File(nbar_fname, "r")

    gal_cube = gal_file["datacube"].value
    nbar_cube = nbar_file["datacube"].value

    gal_file.close()
    nbar_file.close()

    gal_cube = (gal_cube - nbar_cube) / nbar_cube
    # set bad values to 0
    gal_cube[np.logical_not(np.isfinite(gal_cube))] = 0.

    return gal_cube


def cube_properties(filename):
    cube_file = h5py.File(filename, "r")
    out = {}
    out["z_left"] = cube_file["z_left"].value
    out["z_right"] = cube_file["z_right"].value
    out["z_counts"] = cube_file["z_counts"].value
    out["freq"] = cube_file ["freq"].value
    cube_file.close()

    return out


def write_dict(filename, dict_data):
    data = h5py.File(filename, "w")
    for k, v in dict_data.items():
        data[k] = v

    data.close()


def cube_power(lcube, lw_ang, rcube, rw_ang,
               mixing_matrix=None, lmax=399, deltal=20):
    """
    Find the Cl power along each slice of cubes
    give left and right sides and respective ANGULAR weights
    """
    if mixing_matrix is None:
        mixing_matrix = sm.mixing_from_weight(lw_ang, rw_ang, lmax=lmax)

    nslice = lcube.shape[1]

    cl_set = []
    for index in range(nslice):
        sig = pe.estimate_power(lcube[:, index], lw_ang,
                                rcube[:, index], rw_ang,
                                deltal=deltal, mixing_matrix=mixing_matrix, lmax=lmax)

        binned_ell = sig[0]
        cl_set.append(sig[1])

    nell = binned_ell.size
    clmat = np.zeros(shape=(nell, nslice))
    for index in range(nslice):
        clmat[:, index] = cl_set[index]

    return binned_ell, clmat


def avg_rand(line, tag):
    name = "%s_%s" % (tag, line)
    rand_list = glob.glob("./cubes/*%s_C_ell_rand*" % name)
    outfile = "cubes/PIXIE_%s_shot.h5" % name

    # get the cube properties of a reference power spectrum
    cube_prop = cube_properties(rand_list[0])
    ref_file = h5py.File(rand_list[0], "r")
    clmat_shape = ref_file["C_ell"].value.shape
    ell = ref_file["ell"].value
    ref_file.close()
    nrand = len(rand_list)
    clmat_comp = np.zeros(shape=(nrand, clmat_shape[0], clmat_shape[1]))

    for index, rand_file in enumerate(rand_list):
        print(index, rand_file)
        rand = h5py.File(rand_file, "r")
        clmat_comp[index, :, :] = rand["C_ell"].value
        rand.close()

    cube_prop["C_ell_all"] = clmat_comp
    cube_prop["C_ell"] = np.nanmean(clmat_comp, axis=0)
    cube_prop["C_ell_err"] = np.nanstd(clmat_comp, axis=0)
    cube_prop["ell"] = ell
    write_dict(outfile, cube_prop)


def find_power(line, tag, lmax=399):
    name = "%s_%s" % (tag, line)
    rand_list = glob.glob("./cubes/*%s_rand*" % name)

    nbar_file = "cubes/PIXIE_%s_nbar.h5" % name
    gal_file = "cubes/PIXIE_%s_gal.h5" % name
    outfile = "cubes/PIXIE_%s_C_ell.h5" % name

    print("nbar:", nbar_file)
    print("gal: ",gal_file)
    print("out: ", outfile)
    cube_prop = cube_properties(gal_file)
    print(cube_prop)

    angular_weight = make_ang_weight(nbar_file)
    delta = make_delta(gal_file, nbar_file)
    mixing_matrix = sm.mixing_from_weight(angular_weight, angular_weight,
                                          lmax=lmax)

    ell, clmat = cube_power(delta, angular_weight, delta, angular_weight,
                            mixing_matrix=mixing_matrix, lmax=lmax)

    cube_prop["C_ell"] = clmat
    cube_prop["ell"] = ell
    write_dict(outfile, cube_prop)

    for rand_file in rand_list:
        delta = make_delta(rand_file, nbar_file)
        outfile = "cubes/PIXIE_%s_C_ell_%s" % (name, rand_file.split("_")[-1])
        print(outfile)

        ell, clmat = cube_power(delta, angular_weight, delta, angular_weight,
                                mixing_matrix=mixing_matrix, lmax=lmax)

        cube_prop["C_ell"] = clmat
        cube_prop["ell"] = ell
        write_dict(outfile, cube_prop)


def find_all_power(nu0_lines):
    for line, nu0_line in nu0_lines.items():
        find_power(line, "CMASS_North")
        #find_power(line, "CMASS_South")
        find_power(line, "LOWZ_North")
        #find_power(line, "LOWZ_South")


def avg_all_rand(nu0_lines):
    for line, nu0_line in nu0_lines.items():
        avg_rand(line, "CMASS_North")
        #avg_rand(line, "CMASS_South")
        avg_rand(line, "LOWZ_North")
        #avg_rand(line, "LOWZ_South")


def run_all_lines():
    nu0_lines = {}
    nu0_lines["CII"] = nu0_cii
    nu0_lines["CO1-0"] = co_freq(0)
    nu0_lines["CO2-1"] = co_freq(1)
    nu0_lines["CO3-2"] = co_freq(2)
    nu0_lines["CO4-3"] = co_freq(3)
    nu0_lines["CO5-4"] = co_freq(4)
    nu0_lines["CO6-5"] = co_freq(5)
    nu0_lines["CO7-6"] = co_freq(6)
    nu0_lines["CO8-7"] = co_freq(7)
    nu0_lines["CO9-8"] = co_freq(8)
    nu0_lines["CO10-9"] = co_freq(9)
    nu0_lines["CO11-10"] = co_freq(10)
    nu0_lines["CO12-11"] = co_freq(11)

    #find_all_power(nu0_lines)
    avg_all_rand(nu0_lines)


def make_survey_masks(outfile, survey, region, line):
    """Convert nbar into 1/0 angular mask region
    """
    tag = "%s_%s_%s" % (survey, region, line)
    ang_weight = make_ang_weight("cubes/PIXIE_%s_nbar.h5" % tag)
    mask = np.zeros_like(ang_weight)
    # this should not be needed?
    good_region = ang_weight > 0.01
    mask[good_region] = 1.
    f_sky = np.sum(good_region) / float(ang_weight.shape[0])
    print("%s, f_sky: %105.5g" % (tag, f_sky))

    out = h5py.File(outfile, "w")
    out["mask"] = mask
    out["ang_weight"] = ang_weight
    out.close()


if __name__ == "__main__":
    #run_all_lines()
    make_survey_masks("PIXIE_CMASS_North_ang_weight.h5", "CMASS", "North", "CII")
    make_survey_masks("PIXIE_CMASS_South_ang_weight.h5", "CMASS", "South", "CII")
    make_survey_masks("PIXIE_LOWZ_North_ang_weight.h5", "LOWZ", "North", "CII")
    make_survey_masks("PIXIE_LOWZ_South_ang_weight.h5", "LOWZ", "South", "CII")

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import healpy as hp
import copy
import h5py

def plot_maps(map_array, weights=None, fname = 'test', norm = None, title_str = 'Map, ', index_name = 'index = ', index_value_strs = None,
index_units = '', colorbar_label = 'MJy/sr', max = None, min = None):
    pp = PdfPages(fname)
    plt.clf()
    if type(index_value_strs) == type(None):
        index_value_strs = [str(ind) for ind in range(map_array.shape[0])]
    for ind in range(map_array.shape[0]):
        plot_map = copy.deepcopy(map_array[ind,:])
        if norm == 'log':
            plot_map[plot_map<=0] = np.nan
        if type(weights) != type(None):
            if weights.ndim == 2:
                if weights.shape[0] == 1:
                    weight = weights[0,:]
                else:
                    weight = weights[ind,:]
            else:
                weight = weights
            plot_map[weight == 0] = np.nan
        hp.mollview(plot_map, title = title_str + index_name + str(index_value_strs[ind]) + index_units + '.', norm=norm, unit = colorbar_label, max = max, min = min)
        pp.savefig()
    pp.close()

def plot_cl_vs_ell(ells, cl, cl_sigma_estimate, title, ylabel, xscale = 'log', yscale = 'log', theory_cl = None, theory_ells = None, sn_theory = None, scatter_theory = False):
    neg_bool = cl<0
    pos_bool = cl>0
    plt.errorbar(ells[pos_bool], cl[pos_bool], yerr = cl_sigma_estimate[pos_bool], fmt='o', c='blue', label = 'Positive.')
    plt.errorbar(ells[neg_bool], -cl[neg_bool], yerr = cl_sigma_estimate[neg_bool], fmt='o', c='red', label = 'Negative.')
    if type(theory_cl) != type(None):
        if not scatter_theory:
            if np.min(theory_cl)>0:
                plt.plot(theory_ells, theory_cl, c = 'cyan', label = 'Expected signal, positive.')
        if type(sn_theory) != type(None):
            plt.plot(theory_ells, sn_theory, c = 'green', label = 'Expected shot noise only.')
        if yscale == 'log':
            if np.min(theory_cl)<0:
                plt.plot(theory_ells, -theory_cl, c = 'orange', label = 'Expected signal, negative.')
        else:
            neg_bool = theory_cl<0
            pos_bool = theory_cl>0
            plt.scatter(theory_ells[pos_bool], theory_cl[pos_bool],  marker='x', c='blue', label = 'Positive model.')
            plt.scatter(theory_ells[neg_bool], -theory_cl[neg_bool], marker='x', c='red', label = 'Negative model.')
    plt.yscale(yscale)
    plt.xscale(xscale)
    plt.xlabel(r'$\ell$')
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)

def plot_cl_zpairs_vs_ell(tom_pair, cl_zz_sigma_estimate, z_inds1, z_inds2, z_labels1, z_labels2, title = '', fname = 'test', units = None, all_pairs=True, xscale = 'log', yscale = 'log', include_model = False, add_sn_to_theory = None, scatter_theory = False):
    pp = PdfPages(fname)
    plt.clf()
    if tom_pair.is_Dell:
        y_title_string = r'$D_{\ell}$'
    else:
        y_title_string = r'$C_{\ell}$'
    y_unit_string = y_title_string
    if type(units) != type(None):
        y_unit_string += '(' + units + ')'
    if include_model:
        #Assume model path is the following.
        #model_path = '/work2/08946/christoa/stampede2/DIRBExBOSS_analysis/Predicted_Signals/'
        #model_path += 'Cl_DIRBE_eBOSS_CMASS_1.25_to_4.9microns_z_0.20_0.60.h5'
        model_path = '/Users/christoa/DIRBExBOSS_analysis/simulations/'
        model_path += 'Cl_zz_bias1p8_withRSD.h5'
        with h5py.File(model_path, 'r') as f:
            #cl_model_cube = np.array(f['cl_wavelength_redshift'])
            cl_model_cube = np.array(f['cl_zz'])[2:,:,:]
            model_ells = np.array(f['ells'])[2:]
    if all_pairs:
        for ind1, z1 in enumerate(z_inds1):
            for ind2, z2 in enumerate(z_inds2):
                current_title = title + y_title_string + ' at ' + z_labels1[ind1] + ' and ' + z_labels2[ind2] + '.'
                #if include_model and ind1<4 and ind2<2:
                #    plot_cl_vs_ell(tom_pair.ells, tom_pair.cl_zz[:,z1,z2], cl_zz_sigma_estimate[:,z1,z2], current_title, y_unit_string, xscale = xscale, yscale = yscale, theory_cl = cl_model_cube[ind1,ind2,:], theory_ells = model_ells)
                if include_model:
                    if type(add_sn_to_theory) == type(None):
                        plot_cl_vs_ell(tom_pair.ells, tom_pair.cl_zz[:,z1,z2], cl_zz_sigma_estimate[:,z1,z2], current_title, y_unit_string, xscale = xscale, yscale = yscale, theory_cl = cl_model_cube[:,z1,z2], theory_ells = model_ells, scatter_theory = scatter_theory)
                    else:
                        full_theory = cl_model_cube + add_sn_to_theory[2:,:,:]
                        if z1!=z2:
                            plot_cl_vs_ell(tom_pair.ells, tom_pair.cl_zz[:,z1,z2], cl_zz_sigma_estimate[:,z1,z2], current_title, y_unit_string, xscale = xscale, yscale = yscale, theory_cl = full_theory[:,z1,z2], theory_ells = model_ells, scatter_theory = scatter_theory)
                        else:
                            plot_cl_vs_ell(tom_pair.ells, tom_pair.cl_zz[:,z1,z2], cl_zz_sigma_estimate[:,z1,z2], current_title, y_unit_string, xscale = xscale, yscale = yscale, theory_cl = full_theory[:,z1,z2], theory_ells = model_ells, scatter_theory = scatter_theory, sn_theory = add_sn_to_theory[2:,z1,z2])
                else:
                    plot_cl_vs_ell(tom_pair.ells, tom_pair.cl_zz[:,z1,z2], cl_zz_sigma_estimate[:,z1,z2], current_title, y_unit_string, xscale = xscale, yscale = yscale)
                plt.tight_layout()
                pp.savefig()
                plt.show()
    else:
        for ind1, z1 in enumerate(z_inds1):
            current_title = title + y_title_string + ' at ' + z_labels1[ind1] + ' and ' + z_labels1[ind1] + '.'
            plot_cl_vs_ell(tom_pair.ells, tom_pair.cl_zz[:,z1,z1], cl_zz_sigma_estimate[:,z1,z1], current_title, y_unit_string, xscale = xscale, yscale = yscale)
            plt.tight_layout()
            pp.savefig()
            plt.show()
    pp.close()

def plot_cl_zz_cube(zz_cube, z1_centers = None, z2_centers = None, title = None, units = None, xlabel = None, ylabel = None, corr = False):
    if not corr:
        plt.imshow(zz_cube)
    else:
        means = np.diagonal(zz_cube)**0.5
        means_arr = means[:,None]*means[None,:]
        plt.imshow(zz_cube/means_arr)
    if type(z1_centers) != type(None):
        plt.yticks(np.arange(zz_cube.shape[0]), z1_centers)
    if type(z2_centers) != type(None):
        plt.xticks(np.arange(zz_cube.shape[1]), z2_centers)
    cbar = plt.colorbar()
    cbar.set_label(units)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_cl_zz_cubes(tom_pair, ell_inds = None, z1_centers = None, z2_centers = None, title = '', fname = 'test', units = None, xlabel = None, ylabel = None, corr = False):
    pp = PdfPages(fname)
    plt.clf()
    if tom_pair.is_Dell:
        y_title_string = r'$D_{\ell}$'
    else:
        y_title_string = r'$C_{\ell}$'
    if type(ell_inds) == type(None):
        ell_inds = np.arange(tom_pair.cl_zz.shape[0])
    for ind in ell_inds:
        this_title = title + ' ' + y_title_string + ' at $\ell$=' + str(round(tom_pair.ells[ind])) + '.'
        plot_cl_zz_cube(tom_pair.cl_zz[ind,:,:], z1_centers = z1_centers, z2_centers = z2_centers, title = this_title, units = units, xlabel = xlabel, ylabel=ylabel, corr = corr)
        plt.tight_layout()
        pp.savefig()
        plt.show()
    pp.close()

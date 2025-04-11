#Module containing Cl(z,z) functions of cosmological/astrophysical parameters.
#For FIRASxBOSS, most of these will assume a cl_model, computed by CLASS, and fit for astrophysical parameters.

import numpy as np
import scipy as sp
from . import thermodynamic
import astropy.units as u
from . import pwrspec_estimation as pe
import copy
from astropy.convolution import convolve
import h5py
import healpy

def realize_maps_from_cl(cl_model, nside=128, return_cl_also = False, window_func = None):
    alms, cl = pe.full_correlated_synfast(cl_model)
    maps = [healpy.alm2map(np.array(alms[:,i]), nside, lmax = 3*nside - 1) for i in range(alms.shape[1])]
    if type(window_func) != type(None):
        window_maps = [healpy.sphtfunc.smoothing(healpy.alm2map(np.array(alms[:,i]),
                                                               nside, lmax = 3*nside - 1), beam_window = window_func) 
                                       for i in range(alms.shape[1])]
    if type(window_func) == type(None):
        ans = np.array(maps)
    else:
        ans = [np.array(maps), np.array(window_maps)]
    if not return_cl_also:
        return ans
    else:
        return ans, cl

#Gives a quantity proportional to the effective CIB emitting mass of a given galaxy as a function of halo mass.
#mass_eff is the halo mass that is most efficient at hosting star formation; sigma_lm defines width of efficient star formation.
#mass_n is a normalization factor, which I have arbitrairly set to 1.
#Reference DOI: 10.1093/mnras/sty1243
def sigma(mass):
    sigma_lm = 0.5
    mass_eff = 10.**12.6
    mass_n = 1.
    ans=(mass/mass_n)*((2*np.pi*sigma_lm**2.)**-0.5)*np.exp(-(np.log10(mass)-np.log10(mass_eff))**2./(2*sigma_lm**2.))
    return ans

#This function integrates the effective star forming mass at redshift z, integrating from halo mass 10**10 to 10**15.
def int_sigma(z):
    from hmf import MassFunction
    import scipy.integrate
    hmf = MassFunction()
    mass_func = hmf.dndm
    hmf.update(z=z)
    masses = np.logspace(10,15, num=500)
    mass_func = hmf.dndm
    def mass_func_interp(mass_vector):
        return np.interp(mass_vector, masses, mass_func)
    def sigma_times_mf(mass_vector):
        return sigma(mass_vector)*mass_func_interp(mass_vector)
    return scipy.integrate.quad(sigma_times_mf, 10.**10, 10.**15)[0]

#This function vectorizes the int_sigma() function above.
def int_sigmas(z_vect):
    int_sigmas = np.zeros(z_vect.shape)
    for k,z in enumerate(z_vect):
        int_sigmas[k] = int_sigma(z)
    return int_sigmas

#Dust SED function.
#Normalized to amp at lowest frequency.
#The turnover point actually depends on temperature, calculated via log-derivative matching.
#For 26K CIB T_dust (see arXiv 1404.1933), this turnover is around 3.5 THz.
#See astro-ph/0209450.
def spect(f, amp, temp, beta=1.5, turnover=False, temp_derivative = False, fix_normalization = False):
    f = f *u.GHz
    #f_0 = 1500.*u.GHz
    f_0 = 3500.*u.GHz
    if not temp_derivative:
        planck_func = thermodynamic.planck(f, temp*u.Kelvin).to(u.MJy / u.sr)
        #output_unit = u.MJy/u.sr
    else:
        planck_func = thermodynamic.dplanck_dtemp(f_0, temperature = temp* u.Kelvin).to(u.MJy / (u.sr*u.Kelvin))
        #output_unit = u.MJy/(u.sr*u.Kelvin)
    #f = f *u.GHz
    #f_0 = 1500.*u.GHz
    #f_0 = 3500.*u.GHz
    def func1(f):
        #return ((f/f_0)**1.5)*thermodynamic.planck(f, temp* u.Kelvin).to(u.MJy / u.sr)
        #return ((f/f_0)**1.5)*planck_func(f, temp* u.Kelvin).to(output_unit)
        return ((f/f_0)**1.5)*planck_func
    def func2(f):
        #return (f/f_0)**-2*thermodynamic.planck(f_0, temp* u.Kelvin).to(u.MJy / u.sr)
        #return (f/f_0)**-2*planck_func(f_0, temp* u.Kelvin).to(output_unit)
        return (f/f_0)**-2*planck_func
    ans = np.zeros(f.shape)
    if turnover:
        ans[f<f_0] = func1(f[f<f_0])
        ans[f>=f_0] = func2(f[f>=f_0])
    else:
        ans = func1(f)
    if not fix_normalization:
        return amp*ans/ans[0]
    else:
        return amp*ans/ans.flat[0]


#Applies beam/pixel window function (dim: z,ell) to unbinned cl model (dim: ell,z,z).
#Then mixes ells of cl model with mixing matrix.
#Then bins cl model according to start_ell=bin_params[0] and delta_ell = bin_params[1]
def forward_model(unbinned_cl, mixing_matrix, window1, window2, bin_params, ell_vec):
    unbinned_cl = unbinned_cl*window1.T[:,:,None]*window2.T[:,None,:]
    unbinned_cl = np.einsum('ij,jkl', mixing_matrix, unbinned_cl)
    binned_ell, binned_cl = pe.bin_power(ell_vec[bin_params[0]:], unbinned_cl[bin_params[0]:,:] ,bin_params[1], ell_axis = 0)
    return binned_cl

def sn_model_from_sel_func(sel_func):
    nside = (sel_func.shape[-1]/12)**0.5
    if sel_func.ndim != 2:
        raise TypeError('Selection function should be 2-dimensional array [redshift, angle].')
    if nside != int(nside):
        raise TypeError('Selection function should have trailing dimension 12*nside^2.')
    nside = int(nside)
    noise_model = (4.*np.pi/(12*nside**2))*(np.sum(sel_func, axis=-1)/np.sum(sel_func!=0, axis=-1))**-1
    noise_model = np.diag(noise_model)[None,:,:]*np.ones((3*nside))[:,None,None]
    return noise_model

def sn_model_from_cat_and_sel_func(cat, sel_func):
    nside = (sel_func.shape[-1]/12)**0.5
    if sel_func.ndim != 2:
        raise TypeError('Selection function should be 2-dimensional array [redshift, angle].')
    if nside != int(nside):
        raise TypeError('Selection function should have trailing dimension 12*nside^2.')
    nside = int(nside)
    noise_model = (4.*np.pi/(12*nside**2))*(np.sum(cat, axis=-1)/np.sum(sel_func!=0, axis=-1))**-1
    noise_model = np.diag(noise_model)[None,:,:]*np.ones((3*nside))[:,None,None]
    return noise_model

def plot_cl_zz_cubes(data, model, ells = None, z_edges = None, fname = 'temp.pdf', same_scale = True):
    if model.shape != data.shape:
        raise ValueError("Model and data must have same shape")
    if type(z_edges) != type(None):
        extent = [z_edges[0], z_edges[-1], z_edges[0], z_edges[-1]]
    else:
        extent = None
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.style.use('/home/cjander8/plotting_files/default.mplstyle')
    with PdfPages(fname) as pdf:
        for ind in range(model.shape[0]):
            vmin = min(np.min(data[ind,:,:]), np.min(model[ind,:,:]))
            vmax = max(np.max(data[ind,:,:]), np.max(model[ind,:,:]))
            if type(ells) != type(None):
                ell_string = ' $\ell$=' + str(int(ells[ind]))
            else:
                ell_string = '.'
            f, axarr = plt.subplots(1,2)
            if same_scale:
                im = axarr[0].imshow(data[ind,:,:], vmin=vmin, vmax=vmax, origin = 'lower', extent = extent)
            else:
                im = axarr[0].imshow(data[ind,:,:], origin = 'lower', extent = extent)
            axarr[0].set_title('Data' + ell_string)
            axarr[0].set(xlabel = 'z', ylabel = "z'")
            axarr[0].label_outer()

            #ax = plt.gca()
            divider = make_axes_locatable(axarr[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            f.colorbar(im, cax=cax)
            #f.colorbar(im, ax=axarr[0])
            if same_scale:
                im=axarr[1].imshow(model[ind,:,:], vmin=vmin, vmax=vmax, origin = 'lower', extent = extent)
            else:
                im=axarr[1].imshow(model[ind,:,:], origin = 'lower', extent = extent)
            axarr[1].set_title('Model' + ell_string)
            axarr[1].set(xlabel = 'z')
            axarr[1].label_outer()

            #ax = plt.gca()
            divider = make_axes_locatable(axarr[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            #f.colorbar(im, ax=axarr[1])
            f.colorbar(im, cax=cax)
            #if type(ells) != type(None):
            #    plt.title('ell ' + str(ells[ind]))
            plt.tight_layout()
            pdf.savefig()

def plot_cl_zz_cubes_columns(data, model, ells = None, z_edges = None, fname = 'temp.pdf', same_scale = True):
    if model.shape != data.shape:
        raise ValueError("Model and data must have same shape")
    if type(z_edges) != type(None):
        extent = [z_edges[0], z_edges[-1], z_edges[0], z_edges[-1]]
    else:
        extent = None
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.style.use('/home/cjander8/plotting_files/default.mplstyle')
    with PdfPages(fname) as pdf:
        for ind in range(model.shape[0]):
            vmin = min(np.min(data[ind,:,:]), np.min(model[ind,:,:]))
            vmax = max(np.max(data[ind,:,:]), np.max(model[ind,:,:]))
            if type(ells) != type(None):
                ell_string = ' $\ell$=' + str(int(ells[ind]))
            else:
                ell_string = '.'
            f, axarr = plt.subplots(2,1)
            if same_scale:
                im = axarr[0].imshow(data[ind,:,:], vmin=vmin, vmax=vmax, origin = 'lower', extent = extent)
            else:
                im = axarr[0].imshow(data[ind,:,:], origin = 'lower', extent = extent)
            axarr[0].set_title('Data' + ell_string)
            axarr[0].set(xlabel = 'z', ylabel = "z'")
            axarr[0].label_outer()

            #ax = plt.gca()
            divider = make_axes_locatable(axarr[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            f.colorbar(im, cax=cax)
            #f.colorbar(im, ax=axarr[0])
            if same_scale:
                im=axarr[1].imshow(model[ind,:,:], vmin=vmin, vmax=vmax, origin = 'lower', extent = extent)
            else:
                im=axarr[1].imshow(model[ind,:,:], origin = 'lower', extent = extent)
            axarr[1].set_title('Model' + ell_string)
            axarr[1].set(xlabel = 'z', ylabel = "z'")
            axarr[1].label_outer()

            #ax = plt.gca()
            divider = make_axes_locatable(axarr[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            #f.colorbar(im, ax=axarr[1])
            f.colorbar(im, cax=cax)
            #if type(ells) != type(None):
            #    plt.title('ell ' + str(ells[ind]))
            #plt.tight_layout()
            pdf.savefig()

def plot_cl_zz_cubes_alone(data, model, ells, z_edges = None, fname = 'temp.pdf', same_scale = True, format = 'eps', titles = True, cbarlabel = None):
    if model.shape != data.shape:
        raise ValueError("Model and data must have same shape")
    if type(z_edges) != type(None):
        extent = [z_edges[0], z_edges[-1], z_edges[0], z_edges[-1]]
    else:
        extent = None
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import ticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.style.use('/home/cjander8/plotting_files/default.mplstyle')
    from cmcrameri import cm
    #matplotlib.rcParams.update({'font.size': 12})
    for ind in range(model.shape[0]):
        vmin = min(np.min(data[ind,:,:]), np.min(model[ind,:,:]))
        vmax = max(np.max(data[ind,:,:]), np.max(model[ind,:,:]))
        if type(ells) != type(None):
            ell_string = ' $\ell$=' + str(int(ells[ind]))
            ell_string2 = '_ell' + str(ells[ind])
        else:
            ell_string = '.'
            ell_string2 = '_ell' + str(ind)
        #plt.figure(figuresize=(2.2,2.2))
        plt.figure(figsize=(3.3,3.3))
        #plt.figure()
        ax = plt.gca()
        if same_scale:
            im = ax.imshow(model[ind,:,:], vmin=vmin, vmax=vmax, origin = 'lower', extent = extent, cmap = cm.batlow)
        else:
            im=ax.imshow(model[ind,:,:], origin = 'lower', extent = extent, cmap = cm.batlow)
        if titles:
            ax.set_title('Model' + ell_string)
        ax.set(xlabel = 'z', ylabel = "z'")
        ax.label_outer()
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        plt.setp(ax.get_xticklabels()[::2], visible=False)
        plt.setp(ax.get_yticklabels()[::2], visible=False)
        #plt.locator_params(nbins=3)
        if type(cbarlabel) != type(None):
            cbar.ax.set_ylabel(cbarlabel)
        if fname != 'temp.pdf':
            plt.savefig(fname + '_Model_' + ell_string2 + '.' + format , format = format, bbox_inches="tight")
        plt.clf()
        #plt.figure(figuresize=(2.2,2.2))
        plt.figure(figsize=(3.3,3.3))
        #plt.figure()
        ax = plt.gca()
        if same_scale:
            im = ax.imshow(data[ind,:,:], vmin=vmin, vmax=vmax, origin = 'lower', extent = extent, cmap = cm.batlow)
        else:
            im=ax.imshow(data[ind,:,:], origin = 'lower', extent = extent, cmap = cm.batlow)
        if titles:
            ax.set_title('Data' + ell_string)
        ax.set(xlabel = 'z', ylabel = "z'")
        ax.label_outer()
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        #cax = divider.append_axes("right", size="5%", pad=0.0)
        cbar = plt.colorbar(im, cax=cax)
        tick_locator = ticker.MaxNLocator(nbins=5)
        cbar.locator = tick_locator
        cbar.update_ticks()
        #plt.tight_layout()
        #plt.gcf().subplots_adjust(right=0.15)
        #plt.locator_params(numticks=3)
        #ax.locator_params(axis = 'x', nbins=3)
        #ax.locator_params(axis = 'y', nbins=3)
        #if type(z_edges) != type(None):
        #    ax.set_xticklabels(np.linspace(z_edges[0], z_edges[-1], num=4))
        #    ax.set_yticklabels(np.linspace(z_edges[0], z_edges[-1], num=4))
        plt.setp(ax.get_xticklabels()[::2], visible=False)
        plt.setp(ax.get_yticklabels()[::2], visible=False)
        #plt.locator_params(nbins=3)
        if type(cbarlabel) != type(None):
            cbar.ax.set_ylabel(cbarlabel)
        if fname != 'temp.pdf':  
            plt.savefig(fname + 'Data_' + ell_string2 + '.' + format, format = format, bbox_inches="tight")
        plt.clf()

def plot_cl_vs_ell_or_z(data, model_dict, covariance, z_index_offset = 0, z_or_ell = 'ell', ells = None, z_edges = None, y_units_string = '', logscale = False, logy=False, legend = True, times_ell_ell1 = False, fname = 'temp.pdf'):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    z_size = data.shape[-1]
    z1_indeces = np.arange(z_size - z_index_offset)
    z2_indeces = z1_indeces + z_index_offset
    plot_data = data[:, z1_indeces, z2_indeces]
    plot_model_dict = {}
    if type(ells) == type(None):
        ells = np.arange(data.shape[0])
    if times_ell_ell1:
        ell_ell1_string = '$\ell(\ell+1)$$'
        plot_data = plot_data*(ells*(ells+1))[:,None]
    else:
        ell_ell1_string = ''
    for key in list(model_dict.keys()):
        plot_model_dict[key] = model_dict[key][:, z1_indeces, z2_indeces]
        if times_ell_ell1:
            plot_model_dict[key] = model_dict[key][:, z1_indeces, z2_indeces]*(ells*(ells+1))[:,None]
    if covariance.ndim == 6:
        error = np.einsum('ijkijk->ijk', covariance)**0.5
        error = error[:, z1_indeces, z2_indeces]
    if covariance.ndim == 5:
        error = np.einsum('ijkjk->ijk', covariance)**0.5
        error = error[:, z1_indeces, z2_indeces]
    if covariance.ndim == 4:
        from . import cl_cov as clc
        cov_indeces = clc.upper_diag_index(z1_indeces,z2_indeces)
        #error = np.einsum('iij->ij', covariance[:,cov_indeces,:,cov_indeces])**0.5
        error = np.einsum('jii->ij', covariance[:,cov_indeces,:,cov_indeces])**0.5
    if covariance.ndim == 3:
        #Already diagonal
        error =  covariance[:, z1_indeces, z2_indeces]**0.5
    with PdfPages(fname) as pdf:
        if(z_or_ell) == 'ell':
            for ind in z1_indeces:
                if type(z_edges) == type(None):
                    z1 = ind
                    z2 = ind + z_index_offset
                else:
                    z1 = (z_edges[ind] + z_edges[ind+1])/2.
                    z2 = (z_edges[ind+z_index_offset] + z_edges[ind+1+z_index_offset])/2.
                f, axarr = plt.subplots(1,1)
                for key in list(plot_model_dict.keys()):
                    plot_model = plot_model_dict[key]
                    plt.plot(ells, plot_model[:,ind], label=key)
                plt.errorbar(ells, plot_data[:,ind], yerr = error[:,ind], fmt='x')
                plt.title('z1=' + str(round(z1, 4)) + ', z2=' + str(round(z2, 4)))
                if logscale:
                    plt.loglog()
                if logy:
                    plt.yscale('log')
                if legend:
                    plt.legend()
                plt.ylabel(ell_ell1_string + '$C_{\ell}$ ' + y_units_string)
                plt.xlabel('$\ell$')
                pdf.savefig()
        else:
            z1s = (z_edges[z1_indeces] + z_edges[z1_indeces + 1])/2.
            z2s = (z_edges[z2_indeces] + z_edges[z2_indeces + 1])/2.
            avg_z_diff = np.mean(z2s -z1s)
            for (ind,ell) in enumerate(ells):
                f, axarr = plt.subplots(1,1)
                for key in list(plot_model_dict.keys()):
                    plot_model = plot_model_dict[key]
                    plt.plot(z1s, plot_model[ind,:], label=key)
                plt.errorbar(z1s, plot_data[ind,:], yerr = error[ind,:], fmt='x')
                if avg_z_diff != 0:
                    plt.title('$\ell$=' + str(int(ell)) + ', z2=z1 + ' + str(round(avg_z_diff, 4)))
                else:
                    plt.title('$\ell$=' + str(int(ell)))
                if logscale:
                    plt.loglog()
                if logy:
                    plt.yscale('log')
                if legend:
                    plt.legend()
                plt.ylabel(ell_ell1_string + '$C_{\ell}$ ' + y_units_string)
                if avg_z_diff != 0:
                    plt.xlabel('$z1$')
                else:
                    plt.xlabel('$z$')
                pdf.savefig()

class Cl_zz_model(object):
    #Basic class for all Cl(z,z') models.
    #Specific model types will inherit basic functions from this Class.
    def save(self, path):
        with h5py.File(path, 'w') as f:
            for key in list(self.__dict__.keys()):
                if type(self.__dict__[key]) == np.ndarray:
                    f[key] = self.__dict__[key]

    def cl(self, params, cl_range):
        #Should be overwritten for each specific model.
        return None

    def set_cl_for_sims(self, params, cl_range):
        self.cl_model_for_sims = self.cl(params, cl_range)

    def realize_alms_ps(self, params, cl_range, win_func = None):
        #try:
        #    cl_model = self.cl_model_for_sims
        #except AttributeError:
            #cl_model = self.set_cl_for_sims(params, cl_range)
            #cl_model = self.cl_model_for_sims
        self.set_cl_for_sims(params, cl_range)
        cl_model = self.cl_model_for_sims
        if type(win_func) != type(None):
            cl_model = cl_model*(win_func[:,None,None])**2
        return pe.full_correlated_synfast(cl_model) 

    def realize_maps(self, params, cl_range, win_func = None, return_cl_also = False):
        import healpy as hp
        alms, cl = self.realize_alms_ps(params, cl_range, win_func = win_func)
        #print alms.shape
        #alm_list = [np.array(alms[:,i]) for i in range(alms.shape[1])]
        #print len(alm_list)
        #print alm_list[0].shape
        maps = [hp.alm2map(np.array(alms[:,i]), int(cl_range[1]/3), lmax = cl_range[1] - 1) for i in range(alms.shape[1])]
        #maps = hp.alm2map(alm_list, cl_range[1]/3, lmax = cl_range[1] - 1)
        #maps = np.array(hp.sphtfunc.alm2map(alm_list, cl_range[1]/3))
        ans = np.array(maps)
        if not return_cl_also:
            return ans
        else:
            return ans, cl
        
 
#Class for galaxy auto-power.
class gal_auto(Cl_zz_model):
    #If forward modeling to pseudo-cls, then these models should have beam/pix window function and mixing applied.
    cl_model = None
    sn_model = None
    shot_fit_per_z = False

    def __init__(self, cl_model, sn_model):
        self.cl_model = cl_model
        self.sn_model = sn_model

    def cl(self, params, cl_range):
        #params[0] is galaxy bias.
        #params[1] is overall shot noise amplitude multiplier if shot_fit_per_z is False.
        #params[1:1+cl_range[1]] are shot noise multipliers per redshift slice if shot_fit_per_z==True.
        model = params[0]**2.*self.cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        if not self.shot_fit_per_z:
            model += params[1]*self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        else:
            model += np.diag(params[1:cl_range[3]-cl_range[2]+1])[None,:,:]*self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        return model

class gal_auto_rsd(Cl_zz_model):
    #Split cl into 3 terms. One is quadratic in bias, 1 is linear, and 1 has no bias factor.
    b2_cl_model = None
    b1_cl_model = None
    b0_cl_model = None
    sn_model = None
    shot_fit_per_z = False

    def __init__(self, b2_cl_model, b1_cl_model, b0_cl_model, sn_model):
        self.b2_cl_model = b2_cl_model
        self.b1_cl_model = b1_cl_model
        self.b0_cl_model = b0_cl_model
        self.sn_model = sn_model

    @classmethod
    def from_clzz_file(cls, path):
        with h5py.File(path) as f:
            return cls(np.array(f['b2_cl_model']), np.array(f['b1_cl_model']), np.array(f['b0_cl_model']), np.array(f['sn_model']))

    @classmethod
    def from_cl_file_and_mapcube(cls, path, mapcube, bin_models = True, forward_model = False, confirm_z_edges = False, simulate_weighted_mean_subt=False, model_z_index_cut = None):
        with h5py.File(path, 'r') as f:
            b2_cl_model = np.array(f['b2_cl_model'])
            b1_cl_model = np.array(f['b1_cl_model'])
            b0_cl_model = np.array(f['b0_cl_model'])
            cl_z_edges = np.array(f['z_edges'])
        if type(model_z_index_cut) != type(None):
            b2_cl_model = b2_cl_model[:,model_z_index_cut[0]:model_z_index_cut[1],model_z_index_cut[0]:model_z_index_cut[1]]
            b1_cl_model = b1_cl_model[:,model_z_index_cut[0]:model_z_index_cut[1],model_z_index_cut[0]:model_z_index_cut[1]]
            b0_cl_model = b0_cl_model[:,model_z_index_cut[0]:model_z_index_cut[1],model_z_index_cut[0]:model_z_index_cut[1]]
            cl_z_edges = cl_z_edges[model_z_index_cut[0]:model_z_index_cut[1]+1]
        try:
            sn_model = sn_model_from_sel_func(mapcube.weights)
        except AttributeError:
            sn_model = sn_model_from_sel_func(mapcube.mapcube2.weights)
        if sn_model.shape != b2_cl_model.shape:
            raise TypeError('Shot noise and Cl model should have the same shape.')
        if confirm_z_edges:
            try:
                map_z_edges = mapcube.z_edges
            except AttributeError:
                map_z_edges = mapcube.mapcube2.z_edges
            if not np.array_equal(cl_z_edges, map_z_edges):
                raise ValueError('MapCube data and Cl model do not have the same redshift edges.')
        if bin_models:
            b2_cl_model = mapcube.bin_cl_model(b2_cl_model, forward_model = forward_model, simulate_weighted_mean_subt = simulate_weighted_mean_subt)
            b1_cl_model = mapcube.bin_cl_model(b1_cl_model, forward_model = forward_model, simulate_weighted_mean_subt = simulate_weighted_mean_subt)
            b0_cl_model = mapcube.bin_cl_model(b0_cl_model, forward_model = forward_model, simulate_weighted_mean_subt = simulate_weighted_mean_subt)
            sn_model = mapcube.bin_cl_model(sn_model, forward_model = forward_model, simulate_weighted_mean_subt = simulate_weighted_mean_subt)
        return cls(b2_cl_model, b1_cl_model, b0_cl_model, sn_model)

    def d_cl_d_params(self, params, cl_range):
        param_len = len(params)
        ans = np.zeros((param_len,cl_range[1] - cl_range[0], cl_range[3] - cl_range[2], cl_range[3] - cl_range[2]))
        ans[0,:] = 2.*params[1]*self.b2_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        ans[0,:] += self.b1_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        if not self.shot_fit_per_z:
            ans[1,:] = self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        else:
            sn_model = self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            ans[1:,:] = np.einsum('jkl,ijm->iklm', np.fill_diagonal(np.zeros((param_len-1, param_len-1, param_len-1)), 1.), sn_model)
            #ans[1:,:] += np.*self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        return ans

    def cl(self, params, cl_range):
        #params[0] is galaxy bias.
        #params[1] is overall shot noise amplitude multiplier if shot_fit_per_z is False.
        #params[1:1+cl_range[1]] are shot noise multipliers per redshift slice if shot_fit_per_z==True.
        model = params[0]**2.*self.b2_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        model += params[0]*self.b1_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        model += self.b0_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        if not self.shot_fit_per_z:
            model += params[1]*self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        else:
            model += np.diag(params[1:cl_range[3]-cl_range[2]+1])[None,:,:]*self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        return model

class gal_auto_rsd_linear_bias_evolution(Cl_zz_model):
    #Split cl into 3 terms. One is quadratic in bias, 1 is linear, and 1 has no bias factor.
    b2_cl_model = None
    b1_cl_model = None
    b0_cl_model = None
    sn_model = None
    shot_fit_per_z = False

    def __init__(self, b2_cl_model, b1_cl_model, b0_cl_model, sn_model, z_edges):
        self.b2_cl_model = b2_cl_model
        self.b1_cl_model = b1_cl_model
        self.b0_cl_model = b0_cl_model
        self.sn_model = sn_model
        self.z_edges = z_edges
        self.z_mean = (z_edges[1:] + z_edges[:-1])/2.
        self.delta_z = self.z_mean - self.z_mean[0]

    @classmethod
    def from_clzz_file(cls, path):
        with h5py.File(path) as f:
            return cls(np.array(f['b2_cl_model']), np.array(f['b1_cl_model']), np.array(f['b0_cl_model']), np.array(f['sn_model']), np.array(f['z_edges']))

    @classmethod
    def from_cl_file_and_mapcube(cls, path, mapcube, bin_models = True, forward_model = False, confirm_z_edges = False, simulate_weighted_mean_subt=False):
        with h5py.File(path, 'r') as f:
            b2_cl_model = np.array(f['b2_cl_model'])
            b1_cl_model = np.array(f['b1_cl_model'])
            b0_cl_model = np.array(f['b0_cl_model'])
            cl_z_edges = np.array(f['z_edges'])
        try:
            sn_model = sn_model_from_sel_func(mapcube.weights)
        except AttributeError:
            sn_model = sn_model_from_sel_func(mapcube.mapcube2.weights)
        if sn_model.shape != b2_cl_model.shape:
            raise TypeError('Shot noise and Cl model should have the same shape.')
        if confirm_z_edges:
            try:
                map_z_edges = mapcube.z_edges
            except AttributeError:
                map_z_edges = mapcube.mapcube2.z_edges
            if not np.array_equal(cl_z_edges, map_z_edges):
                raise ValueError('MapCube data and Cl model do not have the same redshift edges.')
        if bin_models:
            b2_cl_model = mapcube.bin_cl_model(b2_cl_model, forward_model = forward_model, simulate_weighted_mean_subt = simulate_weighted_mean_subt)
            b1_cl_model = mapcube.bin_cl_model(b1_cl_model, forward_model = forward_model, simulate_weighted_mean_subt = simulate_weighted_mean_subt)
            b0_cl_model = mapcube.bin_cl_model(b0_cl_model, forward_model = forward_model, simulate_weighted_mean_subt = simulate_weighted_mean_subt)
            sn_model = mapcube.bin_cl_model(sn_model, forward_model = forward_model, simulate_weighted_mean_subt = simulate_weighted_mean_subt)
        return cls(b2_cl_model, b1_cl_model, b0_cl_model, sn_model, cl_z_edges)

    def d_cl_d_params(self, params, cl_range):
        param_len = len(params)
        ans = np.zeros((param_len,cl_range[1] - cl_range[0], cl_range[3] - cl_range[2], cl_range[3] - cl_range[2]))
        ans[0,:] = 2.*params[1]*self.b2_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        ans[0,:] += self.b1_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        if not self.shot_fit_per_z:
            ans[1,:] = self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        else:
            sn_model = self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            ans[1:,:] = np.einsum('jkl,ijm->iklm', np.fill_diagonal(np.zeros((param_len-1, param_len-1, param_len-1)), 1.), sn_model)
            #ans[1:,:] += np.*self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        return ans

    def cl(self, params, cl_range):
        #params[0] is galaxy bias at redshift index 0.
        #params[1] is galaxy bias slope
        #params[2] is overall shot noise amplitude multiplier if shot_fit_per_z is False.
        #params[2:2+cl_range[1]] are shot noise multipliers per redshift slice if shot_fit_per_z==True.
        bias_vector = params[0] + params[1]*self.delta_z
        model = bias_vector[:,None]*bias_vector[None,:]*self.b2_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        model += (0.5*bias_vector[:,None] + 0.5*bias_vector[None,:])*self.b1_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        model += self.b0_cl_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        if not self.shot_fit_per_z:
            model += params[2]*self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        else:
            model += np.diag(params[2:cl_range[3]-cl_range[2]+2])[None,:,:]*self.sn_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        return model



#Milky way auto with custom rank 1 vectors for Galaxy SED.
#The idea is to take the largest SVD mode as the SED.
#class milky_way_auto_empirical():


class milky_way_auto2(Cl_zz_model):

    power_law_pcl_correction = None
    thermal_pcl_correction = None

    def __init__(self, thermal_noise_cl_unbinned, cubepair, bin_models = True, forward_model = False, a_vect=None, apply_win_func_unbinned = False, simulate_weighted_mean_subt=False, power_law_pcl_correction = None, thermal_pcl_correction = None):
        self.bin_models = bin_models
        self.forward_model = forward_model
        self.cubepair = cubepair
        self.freqs = cubepair.mapcube1.frequencies
        self.ell_vect = np.arange(cubepair.mapcube1.nside*3)
        self.ell_vect[0] = 1.
        self.turnover = False
        self.apply_win_func_unbinned = apply_win_func_unbinned
        self.simulate_weighted_mean_subt = simulate_weighted_mean_subt
        self.define_weighted_mean_subt_correction(power_law_pcl_correction, thermal_pcl_correction)
        if self.bin_models:
            self.thermal_noise = cubepair.bin_cl_model(thermal_noise_cl_unbinned, forward_model = forward_model, thermal_noise = True, simulate_weighted_mean_subt = simulate_weighted_mean_subt, pcl_correction = self.thermal_pcl_correction)
        else:
            #self.thermal_noise = thermal_noise_cl_unbinned*cubepair.thermal_transfer_func[:,None,None]
            self.thermal_noise = thermal_noise_cl_unbinned*cubepair.thermal_transfer_func[:,None,None]
        if type(a_vect) != type(None):
            if len(a_vect) > self.thermal_noise.shape[-1]:
                a_vect = a_vect[:self.thermal_noise.shape[-1]]
            elif len(a_vect) < self.thermal_noise.shape[-1]:
                new_vect = np.zeros((self.thermal_noise.shape[-1]))
                new_vect[:len(a_vect)] = a_vect
                a_vect = new_vect
            self.apply_a_vector_to_noise(a_vect)
        #self.power_law_pcl_correction = None
        #self.thermal_pcl_correction = None

    def define_weighted_mean_subt_correction(self, power_law_pcl_correction, thermal_pcl_correction = None):
        self.power_law_pcl_correction = power_law_pcl_correction
        self.thermal_pcl_correction = thermal_pcl_correction

    @classmethod
    def thermal_noise_from_weights(cls, cubepair, bin_models = True, forward_model = False,  a_vect=None, apply_win_func_unbinned = False, simulate_weighted_mean_subt=False, power_law_pcl_correction = None, thermal_pcl_correction = None):
       #noise_expected = cubepair.mapcube1.weights**-1
       #noise_expected[cubepair.mapcube1.weights==0]=0
       #Computing inverse noise weighted noise avg.
       noise_avg = np.sum(cubepair.mapcube1.weights!=0, axis = -1)/np.sum(cubepair.mapcube1.weights, axis = -1)
       thermal_noise_model = (4.*np.pi/(12*cubepair.nside**2))*np.diag(noise_avg)[None,:,:]*np.ones((3*cubepair.nside))[:,None,None]
       return cls(thermal_noise_model, cubepair, bin_models = bin_models, forward_model = forward_model,  a_vect=a_vect, apply_win_func_unbinned = apply_win_func_unbinned, simulate_weighted_mean_subt=simulate_weighted_mean_subt, power_law_pcl_correction = power_law_pcl_correction, thermal_pcl_correction = thermal_pcl_correction)
        

    def apply_a_vector_to_noise(self, a_vector):
        self.apply_a_to_noise = True
        n_vect = np.einsum('ijj->ij', self.thermal_noise**0.5)
        z_mat = np.arange(self.thermal_noise.shape[-1])
        z_mat = np.abs(z_mat[:,None] - z_mat[None,:])
        self.thermal_noise = a_vector[z_mat][None,:,:]*n_vect[:,:,None]*n_vect[:,None,:]

    def d_cl_d_params(self, params, cl_range):
        #params[0] is Galaxtic emission power spectrum index in ell.
        #params[1] is the amplitude multiplying the Galactic emission SED (at ell=1).
        #params[2] is the temperature of the grey-body model for Galactic emission.
        #params[3] is the thermal_noise amplitude multiplier.
        ell_size = cl_range[1] - cl_range[0]
        z_size = cl_range[3] - cl_range[2]
        ans = np.zeros((4, ell_size, z_size, z_size))
        fish0 = [params[0], params[1], params[2], 0]
        ans[0,:] = self.cl(fish0, cl_range)
        if self.bin_models:
            logs = np.log(self.ell_vect)
            ans[0,:] *= self.cubepair.bin_cl_model_ell_only(logs, forward_model = self.forward_model, simulate_weighted_mean_subt = self.simulate_weighted_mean_subt)[cl_range[0]:cl_range[1],None,None]
        else:
            ans[0,:] *= (np.log(self.ell_vect))[cl_range[0]:cl_range[1],None,None]
        fish1 = [params[0], (2*params[1])**0.5, params[2], 0]
        ans[1,:] = self.cl(fish1, cl_range)
        fish3 = [0., 0., 20., 1.]
        ans[3,:] = self.cl(fish3, cl_range)
        sed = spect(self.freqs, params[1], params[2], turnover=self.turnover)
        sed_deriv = spect(self.freqs, params[1], params[2], turnover=self.turnover, temp_derivative = True)
        if self.bin_models:
            binned_mixed_cl = self.cubepair.bin_cl_model_ell_only(self.ell_vect**params[0], forward_model = self.forward_model, simulate_weighted_mean_subt = self.simulate_weighted_mean_subt)[:,None,None]*(sed[None,:,None]*sed_deriv[None,None,:] + sed_deriv[None,:,None]*sed[None,None,:])
        else:
            binned_mixed_cl = (self.ell_vect**params[0])*(sed[None,:,None]*sed_deriv[None,None,:] + sed_deriv[None,:,None]*sed[None,None,:])
            if self.apply_win_func_unbinned:
                #This currently only works for 1-d window functions. General case is slightly uglier.
                binned_mixed_cl = binned_mixed_cl*self.cubepair.mapcube1.win_func[:,None,None]*self.cubepair.mapcube1.win_func[:,None,None]
        ans[2,:] = binned_mixed_cl[cl_range[0]:cl_range[1], cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
        return ans

    def cl(self, params, cl_range):
        #SED model will be dust greybody with amplitude and temperature parameters.
        #params[0] is Galaxtic emission power spectrum index in ell.
        #params[1] is the amplitude multiplying the Galactic emission SED (at ell=1).
        #params[2] is the temperature of the grey-body model for Galactic emission.
        #params[3] is the thermal_noise amplitude multiplier.
        sed = spect(self.freqs, params[1], params[2], turnover=self.turnover)
        unbinned_unmixed_cl = self.ell_vect**params[0]
        unbinned_unmixed_cl = unbinned_unmixed_cl[:,None,None]*sed[None,:,None]*sed[None,None,:]
        if self.bin_models:
            #binned_mixed_cl = self.cubepair.bin_cl_model(unbinned_unmixed_cl, forward_model = self.forward_model, thermal_noise = False)
            #This form is assuming separability of weights and a window function that only depends on ell.
            if type(self.power_law_pcl_correction) != type(None):
                index = np.argmin(np.abs(params[0] - self.power_law_pcl_correction[:,30]))
                pcl_correction = self.power_law_pcl_correction[index, 1:]
            else:
                pcl_correction = None
            binned_mixed_cl = self.cubepair.bin_cl_model_ell_only(self.ell_vect**params[0], forward_model = self.forward_model, simulate_weighted_mean_subt = self.simulate_weighted_mean_subt, pcl_correction = pcl_correction)[:,None,None]*sed[None,:,None]*sed[None,None,:]
        else:
            binned_mixed_cl = unbinned_unmixed_cl
            if self.apply_win_func_unbinned:
                #This currently only works for 1-d window functions. General case is slightly uglier.
                binned_mixed_cl = binned_mixed_cl*self.cubepair.mapcube1.win_func[:,None,None]*self.cubepair.mapcube1.win_func[:,None,None]
        binned_mixed_cl += params[3]*self.thermal_noise
        return binned_mixed_cl[cl_range[0]:cl_range[1], cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]

class milky_way_auto_2parameters():
    def __init__(self, galaxy, thermal_noise):
        self.thermal_noise = thermal_noise
        self.galaxy = galaxy
    def cl(self, params, cl_range):
        #params[0] is the galaxy amplitude.
        #params[1] is the thermal noise amplitude.
        answer = params[0]*self.galaxy[cl_range[0]:cl_range[1], cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
        answer += params[1]*self.thermal_noise[cl_range[0]:cl_range[1], cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
        return answer

class milky_way_auto_2parameters_per_ell():
    def __init__(self, galaxy, thermal_noise):
        self.thermal_noise = thermal_noise
        self.galaxy = galaxy
    def cl(self, params, cl_range):
        #params[0] is the galaxy amplitude at the first ell index.
        #params[1] is the thermal noise amplitude at the first ell index.
        #Next params index through the galaxy and noise amplitude for the remaining ell indexes.
        #answer = params[0]*self.galaxy[cl_range[0]:cl_range[1], cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
        #answer += params[1]*self.thermal_noise[cl_range[0]:cl_range[1], cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
        #if cl_range[1] - cl_range[0]>1:
        answer = np.zeros((cl_range[1] - cl_range[0], cl_range[3] - cl_range[2], cl_range[3] - cl_range[2]))
        for ind in range(cl_range[0], cl_range[1]):
            answer[ind - cl_range[0],:] = params[0+2*(ind - cl_range[0])]*self.galaxy[ind, cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
            answer[ind - cl_range[0],:] += params[1+2*(ind - cl_range[0])]*self.thermal_noise[ind, cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
        return answer

    def d_cl_d_params(self, params, cl_range):
        answer = np.zeros((cl_range[1] - cl_range[0], cl_range[3] - cl_range[2], cl_range[3] - cl_range[2], len(params)))
        for ind in range(cl_range[0], cl_range[1]):
            answer[ind - cl_range[0],:,:,2*(ind - cl_range[0])] = self.galaxy[ind, cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
            answer[ind - cl_range[0],:,:,2*(ind - cl_range[0])+1] = self.thermal_noise[ind, cl_range[2]:cl_range[3], cl_range[2]:cl_range[3]]
        return answer

#Class for milky_way auto-power at frequencies near 1.5 THz.
class milky_way_auto():
    #Array of consecutive ell-vectors to be considered. For healpy map, it should be np.arange(3*nside).
    ell_vect = None
    #Pixel and beam window function. Dimensions: (z,ell). If data is beam corrected, use all ones.
    window_func = None
    #ell-ell' mixing matrix. Use identity if data is unmixed.
    mixing_matrix = None
    #Length 2 array: bin_params[0] is the ell_vect index to start binning, and bin_params[1] is the bin size.
    bin_params = None
    #Frequencies corresponding to the z-bins.
    freqs = None
    #Thermal noise model, cl(ell,z,z').
    thermal_noise = None
    #Noise window function. If noise in not premixed, use this window function.
    noise_window_func = None
    #Boolean for whether or not to convolve the model by an instrumental frequency response.
    freq_conv = False
    #Boolean for whether to apply a_vector to noise only.
    appy_a_to_noise = False
    #Boolean for whether to use namaster object to do mixing and unmixing.
    namaster_mix = False
    #Boolean controlling whether there is a turnover in dust spectrum. Default is False.
    turnover = False
    #Boolean contorlling whether to use empirical seds form instead of thermal dust.
    use_empirical_seds = False

    def __init__(self, ell_vect, window_func, mixing_matrix, bin_params, freqs, thermal_noise, noise_premixed=True, noise_window_func = None, turnover=False):
        #To avoid dividing by 0, set ell_vect 0 to 1.
        ell_vect[ell_vect==0]=1
        self.ell_vect = ell_vect
        self.window_func = window_func
        self.mixing_matrix = mixing_matrix
        self.bin_params = bin_params
        self.freqs = freqs
        self.turnover = turnover
        self.use_empirical_seds = False
        if type(noise_window_func) == type(None):
            self.noise_window_func = window_func
        else:
            self.noise_window_func = noise_window_func
        if noise_premixed:
            #Option to use if thermal_noise already mixed (with identity if not forward-modeling), window function applied, and binned.
            self.thermal_noise = thermal_noise
        else:
            #Apply noise_window_func, mixing, and binning to unmixed thermal noise model.
            #mixed_thermal_noise = np.einsum('ij,jkl', mixing_matrix, thermal_noise*window_func.T[:,:,None]*window_func.T[:,None,:])
            #binned_ell, self.thermal_noise = pe.bin_power(ell_vect[bin_params[0:], mixed_thermal_noise[bin_params[0]:,:,:], bin_params[1], ell_axis=0)
            self.thermal_noise = forward_model(thermal_noise, mixing_matrix, self.noise_window_func, self.noise_window_func, bin_params, ell_vect)

    def set_freq_conv_mat(self, freq_conv_mat=None):
       #freq_conv_mat: 2d, axis 0 is the final frequency, axis 1 is the input frequency of the unconvolved map or Cl.
       if type(freq_conv_mat) != type(None):
           self.freq_conv = True
           self.freq_conv_mat = freq_conv_mat

    def apply_a_vector_to_noise(self, a_vector):
        self.apply_a_to_noise = True
        n_vect = np.einsum('ijj->ij', self.thermal_noise**0.5)
        z_mat = np.arange(self.thermal_noise.shape[-1])
        z_mat = np.abs(z_mat[:,None] - z_mat[None,:])
        #print z_mat
        #print a_vector[z_mat]
        self.thermal_noise = a_vector[z_mat][None,:,:]*n_vect[:,:,None]*n_vect[:,None,:]

    def use_empirical_seds(self, sed_mat):
        #sed_mat should be n_ell by n_z matrix, with empirical SED's from SVD of Cl data.
        self.use_empirical_seds = True
        self.sed_mat = sed_mat

    def use_namaster_mix(self, namaster_coupling_mat, apply_to_noise=False):
        self.namaster_mix = True
        self.namaster_coupling_mat = namaster_coupling_mat

    def compute_cov_pieces(self, ell_ind, namaster_cov_obj, cl_range):
        #Assumes window_func does not depend on z, assumes both noise and mw signal are unmixed and unbinned.
        import pymaster as nmt
        ell_dep_mw = self.ell_vect**ell_ind
        print((ell_dep_mw.shape))
        print((self.window_func.shape))
        ell_dep_mw *= (self.window_func[0,:]**2.)
        ell_dep_noise = self.thermal_noise[:,0,0]/self.thermal_noise[0,0,0]
        zeros = np.zeros(ell_dep_mw.shape)
        self.mw_mw = nmt.gaussian_covariance(namaster_cov_obj, ell_dep_mw, zeros, zeros, ell_dep_mw)[cl_range[0]:cl_range[1],cl_range[0]:cl_range[1]]
        self.n_n = nmt.gaussian_covariance(namaster_cov_obj, ell_dep_noise, zeros, zeros, ell_dep_noise)[cl_range[0]:cl_range[1],cl_range[0]:cl_range[1]]
        self.mw_n = nmt.gaussian_covariance(namaster_cov_obj, ell_dep_mw, zeros, zeros, ell_dep_noise)[cl_range[0]:cl_range[1],cl_range[0]:cl_range[1]]
        self.thermal_noise_cut = self.thermal_noise[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]

    def finalize_cov(self, params, cl_range):
       if not self.use_empirical_seds:
           sed = spect(self.freqs[cl_range[2]:cl_range[3]], params[1], params[2], turnover=self.turnover)
           frankenstein_cov = params[3]**2*self.n_n[:,None,None,:,None,None]*np.einsum('ik,jn->ijkn', self.thermal_noise_cut[0,:],self.thermal_noise_cut[0,:])[None,:,:,None,:,:]
           frankenstein_cov += params[3]**2*self.n_n[:,None,None,:,None,None]*np.einsum('in,jk->ijkn', self.thermal_noise_cut[0,:],self.thermal_noise_cut[0,:])[None,:,:,None,:,:]
           frankenstein_cov += self.mw_mw[:,None,None,:,None,None]*np.einsum('i,k,j,n->ijkn', sed,sed,sed,sed)[None,:,:,None,:,:]
           frankenstein_cov += self.mw_mw[:,None,None,:,None,None]*np.einsum('i,n,j,k->ijkn', sed,sed,sed,sed)[None,:,:,None,:,:]
           frankenstein_cov += params[3]*self.mw_n[:,None,None,:,None,None]*np.einsum('ik,j,n->ijkn',self.thermal_noise_cut[0,:], sed, sed)[None,:,:,None,:,:]
           frankenstein_cov += params[3]*self.mw_n[:,None,None,:,None,None]*np.einsum('in,j,k->ijkn',self.thermal_noise_cut[0,:], sed, sed)[None,:,:,None,:,:]
           frankenstein_cov += params[3]*self.mw_n[:,None,None,:,None,None]*np.einsum('i,k,jn->ijkn',sed, sed, self.thermal_noise_cut[0,:])[None,:,:,None,:,:]
           frankenstein_cov += params[3]*self.mw_n[:,None,None,:,None,None]*np.einsum('i,n,jk->ijkn', sed, sed, self.thermal_noise_cut[0,:])[None,:,:,None,:,:]
       else:
           sed = self.sed_mat
           frankenstein_cov = params[3]**2*self.n_n[:,None,None,:,None,None]*np.einsum('ik,jn->ijkn', self.thermal_noise_cut[0,:],self.thermal_noise_cut[0,:])[None,:,:,None,:,:]
           frankenstein_cov += params[3]**2*self.n_n[:,None,None,:,None,None]*np.einsum('in,jk->ijkn', self.thermal_noise_cut[0,:],self.thermal_noise_cut[0,:])[None,:,:,None,:,:]
           frankenstein_cov += self.mw_mw[:,None,None,:,None,None]*np.einsum('gi,hk,gj,hn->gijhkn', sed,sed,sed,sed)
           frankenstein_cov += self.mw_mw[:,None,None,:,None,None]*np.einsum('gi,hn,gj,hk->gijhkn', sed,sed,sed,sed)
           frankenstein_cov += params[3]*self.mw_n[:,None,None,:,None,None]*np.einsum('ik,gj,hn->gijhkn',self.thermal_noise_cut[0,:], sed, sed)
           frankenstein_cov += params[3]*self.mw_n[:,None,None,:,None,None]*np.einsum('in,gj,hk->gijhkn',self.thermal_noise_cut[0,:], sed, sed)
           frankenstein_cov += params[3]*self.mw_n[:,None,None,:,None,None]*np.einsum('gi,hk,jn->gijhkn',sed, sed, self.thermal_noise_cut[0,:])
           frankenstein_cov += params[3]*self.mw_n[:,None,None,:,None,None]*np.einsum('gi,hn,jk->gijhkn', sed, sed, self.thermal_noise_cut[0,:])
       return frankenstein_cov 

    def cl(self, params, cl_range):
        #SED model will be dust greybody with amplitude and temperature parameters.
        #parms[0] is Galaxtic emission power spectrum index in ell.
        #parms[1] is the amplitude multiplying the Galactic emission SED (at ell=1).
        #parms[2] is the temperature of the grey-body model for Galactic emission.
        #parms[3] is the thermal_noise amplitude multiplier.
        #sed = spect(self.freqs, params[1], params[2])
        if params[1] != 0 and self.namaster_mix == False:
            sed = spect(self.freqs, params[1], params[2], turnover=self.turnover)
            unbinned_unmixed_cl = self.ell_vect**params[0]
            #Set ell=0 cl to zero.
            unbinned_unmixed_cl[self.ell_vect==0]=0
            #Apply pixel and beam window function.
            unbinned_unmixed_cl = np.einsum('i,ji,ki->ijk', unbinned_unmixed_cl, self.window_func, self.window_func)
            #Apply SED shape in z-direction.
            if not self.use_empirical_seds:
                unbinned_unmixed_cl = unbinned_unmixed_cl*sed[None,:,None]*sed[None,None,:]
            else:
                print((unbinned_unmixed_cl.shape))
                unbinned_unmixed_cl = params[1]**2*unbinned_unmixed_cl[cl_range[0]:cl_range[1],:,:]*self.sed_mat[:,:,None]*self.sed_mat[:,None,:]
            #Apply mixing matrix.
            mixed_cl = np.einsum('ij,jkl', self.mixing_matrix, unbinned_unmixed_cl)
            #Bin the mixed_cl.
            binned_ell, binned_mixed_cl = pe.bin_power(self.ell_vect[self.bin_params[0]:], mixed_cl[self.bin_params[0]:,:,:], self.bin_params[1], ell_axis=0)
            #Cut answer to appropriate cl_range.
            cut_binned_mixed_cl = binned_mixed_cl[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        elif params[1] != 0 and self.namaster_mix == True:
            sed = spect(self.freqs, params[1], params[2], turnover=self.turnover)
            unbinned_unmixed_cl = self.ell_vect**params[0]
            #Set ell=0 cl to zero.
            unbinned_unmixed_cl[self.ell_vect==0]=0
            #Apply NaMaster mixing and binned unmixing.
            #This is currently assuming beam window function does not depend on frequency. Saves computation time.
            unbinned_unmixed_cl *= self.window_func[0,:]**2
            mixed_cl = self.namaster_coupling_mat.couple_cell([unbinned_unmixed_cl])
            binned_unmixed_cl = self.namaster_coupling_mat.decouple_cell(mixed_cl)[0,:]
            #Apply SED shape in z-direction.
            #binned_unmixed_cl = binned_unmixed_cl[:,None,None]*sed[None,:,None]*sed[None,None,:]
            if not self.use_empirical_seds:
                binned_unmixed_cl = binned_unmixed_cl[:,None,None]*sed[None,:,None]*sed[None,None,:]
            else:
                binned_unmixed_cl = params[1]**2*binned_unmixed_cl[cl_range[0]:cl_range[1]][:,None,None]*self.sed_mat[:,:,None]*self.sed_mat[:,None,:] 
            #Cut answer to appropriate cl_range.
            cut_binned_mixed_cl = binned_unmixed_cl[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        #Calculate thermal noise cl.
        noise_cl = params[3]*self.thermal_noise
        cut_noise_cl = noise_cl[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        ans = cut_noise_cl
        if params[1] != 0:
            #print ans.shape
            #print cut_binned_mixed_cl.shape
            ans = ans + cut_binned_mixed_cl
        if self.freq_conv and not self.apply_a_to_noise:
            ans = np.einsum('ij,kjl->kil', self.freq_conv_mat, ans)
            ans = np.einsum('ij, klj->kli', self.freq_conv_mat, ans) 
        return ans
        #return cut_binned_mixed_cl + cut_noise_cl


class cross_power():
    #Moel for Infrared/microwave continuum map crossed with galaxy survey.
    #Example: FIRASxBOSS.
    #Forward modeling here is more complicated because of continuum correlations.
    #The FIRAS and BOSS window functions should be applied after calculating cross-spectrum with non-forward modeled cl_model.
    #Then the cross-power can be forward modeled.
    cl_model = None
    redshifts = None
    freqs = None
    redshift_index = None
    c = 3.*10.**8.
    l_0 = 10.**-18.
    z_indeces_to_keep = None
    freq_conv = False
    conv_freq = False

    def __init__(self, cl_model, redshifts, frequencies, redshift_index, z_indeces_to_keep = None, mixing_matrix = None, window1=None, window2=None, bin_params=None):
        self.cl_model = cl_model
        self.redshifts = redshifts
        self.freqs = frequencies
        self.redshift_index = redshift_index
        if type(z_indeces_to_keep) == type(None):
            #This means that the frequencies and redshifts of the cl model don't extend beyond the normal bounds.
            self.z_indeces_to_keep = [0, self.redshifts.size]
        else:
            #Option to use when cl_model extends beyond redshift range of survey. This can be useful for accurate continuum correlations.
            self.z_indeces_to_keep = z_indeces_to_keep
        if type(mixing_matrix) != type(None) and type(window1) != type(None) and type(window2) != type(None) and type(bin_params) != type(None) and mixing_matrix.shape[-1]==cl_model.shape[0]:
            self.forward_model = True
            self.mixing_matrix = mixing_matrix
            self.window1 = window1
            self.window2 = window2
            self.bin_params = bin_params
            self.ell_vec = np.arange(mixing_matrix.shape[0])
        else:
            self.forward_model = False

    def convolve_freq(self, a_vector):
        self.conv_freq = True
        self.a_vector = a_vector
        z_mat = np.arange(self.cl_model.shape[-1])
        z_mat = np.abs(z_mat[:,None] - z_mat[None,:])
        self.conv_matrix = a_vector[z_mat]

    def precompute(self, make_amps_meaningful=False, fix_line_error = True):
        #Precompute effective star-forming mass per redshift slice.
        self.sigmas = int_sigmas(self.redshifts)
        #Precompute redshift dependence of infra-red luminosity.
        self.phi = (1.+ self.redshifts)**self.redshift_index
        #Precompute Hubble parameter.
        self.H = np.sqrt(0.3*(1.+self.redshifts)**3+0.7)
        self.fix_line_error = fix_line_error
        #If make_amps_meaningful==True, normalize so that line amplitude is the mean line brightness temp and continuum amplitude is the mean continuum brightness temp.
        if make_amps_meaningful:
            self.cont_norm = np.mean(self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)))
            if not fix_line_error:
                self.line_norm = np.mean(self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)**2.))
            else:
                self.line_norm = np.mean(self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H))
        else:
            self.cont_norm = 1.
            self.line_norm = 1.

    def dI_dz_cont(self, params, beta=1.5, matrix=True):
        #parms[0] is the amplitude of the continuum spectrum at 1500 GHz.
        #params[1] is the temperature determining the shape of the continuum spectrum.
        if matrix:
            fz_in = self.freqs[:,None]*(self.redshifts[None,:]+1.)
        else:
            fz_in = self.freqs*(self.redshifts+1.)
        if params[0] !=0:
            theta = spect(fz_in, params[0], params[1], beta=beta)
        else:
            theta = np.zeros(fz_in.shape)
        ans = (theta*self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)))/self.cont_norm
        if self.conv_freq:
            ans = np.einsum('ij,jk', ans, self.conv_matrix)
        return ans

    def dI_dz_line(self, line_factor, matrix=False):
        #line_factor is the line amplitude.
        if not self.fix_line_error:
            ans = (line_factor*self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)**2.))/self.line_norm
        else:
            ans = (line_factor*self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H))/self.line_norm
        if matrix:
            ans = np.diag(ans)
            if self.conv_freq:
                ans = np.einsum('ij,jk', ans, self.conv_matrix)
        return ans

    def cl(self, params, cl_range, beta=1.5):
        #params[0] is the amplitude of the continuum spectrum at 1500 GHz.
        #params[1] is the temperature determining the shape of the continuum spectrum.
        #params[2] is the line amplitude.
        #params[3] is the galaxy bias.
        cont_cross_gal = np.einsum('ij,kjl->kil', self.dI_dz_cont(params[0:2], beta = beta, matrix=True), self.cl_model)
        line_dI_dz = self.dI_dz_line(params[2], matrix = self.conv_freq)
        if not self.conv_freq:
            line_cross_gal = self.dI_dz_line(params[2])[None,:,None]*self.cl_model
        else:
            line_cross_gal = np.einsum('ij,kjl->kil', line_dI_dz, self.cl_model)
        cont_cross_gal = cont_cross_gal[:,self.z_indeces_to_keep[0]:self.z_indeces_to_keep[1],self.z_indeces_to_keep[0]:self.z_indeces_to_keep[1]]
        line_cross_gal = line_cross_gal[:,self.z_indeces_to_keep[0]:self.z_indeces_to_keep[1],self.z_indeces_to_keep[0]:self.z_indeces_to_keep[1]]
        cross_power_uncut = params[3]*(cont_cross_gal + line_cross_gal)
        if self.forward_model == True:
            cross_power_uncut = forward_model(cross_power_uncut, self.mixing_matrix, self.window1, self.window2, self.bin_params, self.ell_vec)
        return cross_power_uncut[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]

    def calc_auto_no_foregrounds(self, params, cl_range, beta=1.5):
        #Calculates what the microwave auto-power would be without foregrounds (so just due to CII and CIB).
        #params[0] is the amplitude of the continuum spectrum at 1500 GHz.
        #parms[1] is the temperature determining the shape of the continuum spectrum.
        #params[2] is the line amplitude.
        cont_spect = self.dI_dz_cont(params[0:2], beta = beta, matrix=True)
        line_spect = self.dI_dz_line(params[2])
        cont_cross_cont = np.einsum('ij,pjk,lk->pil', cont_spect, self.cl_model, cont_spect)
        line_cross_line = self.cl_model*line_spect[None,:,None]*line_spect[None,None,:]
        total_power = cont_cross_cont + line_cross_line
        if self.freq_conv:
            cross_power_uncut = np.einsum('ij,klj->kli', self.freq_conv_mat, np.einsum('ij,kjl->kil', self.freq_conv_mat, total_power)) 
        if self.forward_model == True:
            total_power = forward_model(total_power, self.mixing_matrix, self.window1, self.window2, self.bin_params, self.ell_vec)
        return total_power[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]

class cross_power_rsd():
    #Allow for full linear RSD terms in cross-power. Assumes unit cross-correlation coefficient between optical and Infrared/microwave line map.
    #Moel for Infrared/microwave continuum map crossed with galaxy survey.
    #Example: FIRASxBOSS.
    #Forward modeling here is more complicated because of continuum correlations.
    #The FIRAS and BOSS window functions should be applied after calculating cross-spectrum with non-forward modeled cl_model.
    #Then the cross-power can be forward modeled.

    #The non-RSD component of the Cl spectrum (gets multiplied by galaxy and CII/CIB bias).
    b2_cl_model = None
    #The Cl term that is linear in bias for a galaxy auto-power (RSDxReal). Half of this term will be proportional to CIB/CII bias, and half prop. to galaxy bias.
    b1_cl_model = None
    #The Cl term that is RSDxRSD and therefore has no bias dependence.
    b0_cl_model = None
    redshifts = None
    freqs = None
    redshift_index = None
    c = 3.*10.**8.
    l_0 = 10.**-18.
    z_indeces_to_keep = None
    freq_conv = False
    conv_freq = False

    def __init__(self, b2_cl_model, b1_cl_model, b0_cl_model, redshifts, frequencies, redshift_index, z_indeces_to_keep = None, mixing_matrix = None, window1=None, window2=None, bin_params=None):
        self.b2_cl_model = b2_cl_model
        self.b1_cl_model = b1_cl_model
        self.b0_cl_model = b0_cl_model
        self.redshifts = redshifts
        self.freqs = frequencies
        self.redshift_index = redshift_index
        if type(z_indeces_to_keep) == type(None):
            #This means that the frequencies and redshifts of the cl model don't extend beyond the normal bounds.
            self.z_indeces_to_keep = [0, self.redshifts.size]
        else:
            #Option to use when cl_model extends beyond redshift range of survey. This can be useful for accurate continuum correlations.
            self.z_indeces_to_keep = z_indeces_to_keep
        if type(mixing_matrix) != type(None) and type(window1) != type(None) and type(window2) != type(None) and type(bin_params) != type(None) and mixing_matrix.shape[-1]==cl_model.shape[0]:
            self.forward_model = True
            self.mixing_matrix = mixing_matrix
            self.window1 = window1
            self.window2 = window2
            self.bin_params = bin_params
            self.ell_vec = np.arange(mixing_matrix.shape[0])
        else:
            self.forward_model = False

    def convolve_freq(self, a_vector, edge_comp = False):
        self.conv_freq = True
        self.a_vector = a_vector
        if edge_comp:
            self.a_vector[len(a_vector)/2:] = np.zeros(self.a_vector[len(a_vector)/2:].shape)
        z_mat = np.arange(self.b2_cl_model.shape[-1])
        z_mat = np.abs(z_mat[:,None] - z_mat[None,:])
        self.conv_matrix = a_vector[z_mat]
        if edge_comp:
            for ind in range(self.conv_matrix.shape[0]):
                if ind<len(a_vector):
                    self.conv_matrix[ind,2*ind+1:]*=2.
                elif ind>len(a_vector):
                    self.conv_matrix[ind,:2*ind-self.conv_matrix.shape[1]-1]*=2.
            #norm = np.sum(self.conv_matrix, axis=1)
            #self.conv_matrix /= np.ones(self.conv_matrix.shape)*norm[:,None]
            #self.conv_matrix *= np.mean(norm)

    def precompute(self, make_amps_meaningful=False):
        #Precompute effective star-forming mass per redshift slice.
        self.sigmas = int_sigmas(self.redshifts)
        #Precompute redshift dependence of infra-red luminosity.
        self.phi = (1.+ self.redshifts)**self.redshift_index
        #Precompute Hubble parameter.
        self.H = np.sqrt(0.3*(1.+self.redshifts)**3+0.7)
        #If make_amps_meaningful==True, normalize so that line amplitude is the mean line brightness temp and continuum amplitude is the mean continuum brightness temp.
        if make_amps_meaningful:
            self.cont_norm = np.mean(self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)))
            self.line_norm = np.mean(self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)**2.))
        else:
            self.cont_norm = 1.
            self.line_norm = 1.

    def dI_dz_cont(self, params, beta=1.5, matrix=True):
        #parms[0] is the amplitude of the continuum spectrum at 1500 GHz.
        #params[1] is the temperature determining the shape of the continuum spectrum.
        if matrix:
            fz_in = self.freqs[:,None]*(self.redshifts[None,:]+1.)
        else:
            fz_in = self.freqs*(self.redshifts+1.)
        if params[0] !=0:
            theta = spect(fz_in, params[0], params[1], beta=beta)
        else:
            theta = np.zeros(fz_in.shape)
        ans = (theta*self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)))/self.cont_norm
        if self.conv_freq:
            ans = np.einsum('ij,jk', ans, self.conv_matrix)
        return ans

    def dI_dz_line(self, line_factor, matrix=False):
        #line_factor is the line amplitude.
        ans = (line_factor*self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)**2.))/self.line_norm
        if matrix:
            ans = np.diag(ans)
            if self.conv_freq:
                ans = np.einsum('ij,jk', ans, self.conv_matrix)
        return ans

    def cl(self, params, cl_range, beta=1.5):
        #params[0] is the amplitude of the continuum spectrum at 1500 GHz (times the CII/CIB bias).
        #params[1] is the temperature determining the shape of the continuum spectrum.
        #params[2] is the line amplitude (times the CII/CIB bias).
        #params[3] is the galaxy bias.
        #params[4] is the CII/CIB bias.
        cont_func = self.dI_dz_cont(params[0:2], beta = beta, matrix=True)
        cont_cross_gal = np.einsum('ij,kjl->kil', cont_func, self.b2_cl_model + self.b0_cl_model/(params[3]*params[4]) + self.b1_cl_model/(2.*params[3]) + self.b1_cl_model/(2.*params[4]))
        line_dI_dz = self.dI_dz_line(params[2], matrix = self.conv_freq)
        if not self.conv_freq:
            line_cross_gal = line_dI_dz[None,:,None]*self.b2_cl_model
            line_cross_gal += line_dI_dz[None,:,None]*self.b0_cl_model/(params[3]*params[4])
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[3])
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[4])
        else:
            line_cross_gal = np.einsum('ij,kjl->kil', line_dI_dz, self.b2_cl_model + self.b0_cl_model/(params[3]*params[4]) + self.b1_cl_model/(2.*params[3]) + self.b1_cl_model/(2.*params[4]))
        cont_cross_gal = cont_cross_gal[:,self.z_indeces_to_keep[0]:self.z_indeces_to_keep[1],self.z_indeces_to_keep[0]:self.z_indeces_to_keep[1]]
        line_cross_gal = line_cross_gal[:,self.z_indeces_to_keep[0]:self.z_indeces_to_keep[1],self.z_indeces_to_keep[0]:self.z_indeces_to_keep[1]]
        cross_power_uncut = params[3]*(cont_cross_gal + line_cross_gal)
        if self.forward_model == True:
            cross_power_uncut = forward_model(cross_power_uncut, self.mixing_matrix, self.window1, self.window2, self.bin_params, self.ell_vec)
        return cross_power_uncut[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]

    def calc_auto_no_foregrounds(self, params, cl_range, beta=1.5):
        #Calculates what the microwave auto-power would be without foregrounds (so just due to CII and CIB).
        #params[0] is the amplitude of the continuum spectrum at 1500 GHz (times the CII/CIB bias).
        #parms[1] is the temperature determining the shape of the continuum spectrum.
        #params[2] is the line amplitude (times the CII/CIB bias).
        #params[3] is the CII/CIB bias.
        cont_spect = self.dI_dz_cont(params[0:2], beta = beta, matrix=True)
        line_spect = self.dI_dz_line(params[2])
        #cont_cross_cont = np.einsum('ij,pjk,lk->pil', cont_spect, self.cl_model, cont_spect)

        tot_spect = cont_spect + line_spect
        ans = np.dot(np.dot(tot_spect, self.b2_cl_model), tot_spec.T)
        ans += np.dot(np.dot(tot_spect, self.b0_cl_model), tot_spec.T)/(params[3]**2.)
        ans += np.dot(np.dot(tot_spect, self.b1_cl_model), tot_spec.T)/params[3]

        #line_cross_line = self.cl_model*line_spect[None,:,None]*line_spect[None,None,:]
        #total_power = cont_cross_cont + line_cross_line
        #if self.freq_conv:
        #    cross_power_uncut = np.einsum('ij,klj->kli', self.freq_conv_mat, np.einsum('ij,kjl->kil', self.freq_conv_mat, total_power)) 
        #if self.forward_model == True:
        #    total_power = forward_model(total_power, self.mixing_matrix, self.window1, self.window2, self.bin_params, self.ell_vec)
        #return total_power[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        return ans

class cross_power_rsd2(Cl_zz_model):
    #Allow for full linear RSD terms in cross-power. Assumes unit cross-correlation coefficient between optical and Infrared/microwave line map.
    #Moel for Infrared/microwave continuum map crossed with galaxy survey.
    #Example: FIRASxBOSS.

    #The non-RSD component of the Cl spectrum (gets multiplied by galaxy and CII/CIB bias).
    b2_cl_model = None
    #The Cl term that is linear in bias for a galaxy auto-power (RSDxReal). Half of this term will be proportional to CIB/CII bias, and half prop. to galaxy bias.
    b1_cl_model = None
    #The Cl term that is RSDxRSD and therefore has no bias dependence.
    b0_cl_model = None
    c = 3.*10.**8.
    l_0 = 10.**-18
    conv_freq = False

    def __init__(self, b2_cl_model, b1_cl_model, b0_cl_model, cubepair, bin_cl = True, forward_model = False, redshift_index = 2.3, include_sn = False, simulate_weighted_mean_subt=False):
        if include_sn:
            sn_model = sn_model_from_sel_func(cubepair.mapcube2.weights)
            self.sn_model = sn_model
        if not bin_cl:
            self.b2_cl_model = b2_cl_model
            self.b1_cl_model = b1_cl_model
            self.b0_cl_model = b0_cl_model
        else:
            self.b2_cl_model = cubepair.bin_cl_model(b2_cl_model, forward_model = forward_model, simulate_weighted_mean_subt=simulate_weighted_mean_subt)
            self.b1_cl_model = cubepair.bin_cl_model(b1_cl_model, forward_model = forward_model, simulate_weighted_mean_subt=simulate_weighted_mean_subt)
            self.b0_cl_model = cubepair.bin_cl_model(b0_cl_model, forward_model = forward_model, simulate_weighted_mean_subt=simulate_weighted_mean_subt)
            if include_sn:
                self.sn_model = cubepair.mapcube2.bin_cl_model(sn_model, forward_model = forward_model, simulate_weighted_mean_subt=simulate_weighted_mean_subt)
        self.include_sn = include_sn
        self.cubepair = cubepair
        self.freqs = cubepair.mapcube1.frequencies
        self.bin_cl = bin_cl
        self.forward_model = forward_model
        self.z_edges = cubepair.mapcube1.z_edges
        self.redshifts = (self.z_edges[:-1] + self.z_edges[1:])/2.
        self.dz = np.abs(self.z_edges[:-1] - self.z_edges[1:])
        self.redshift_index = 2.3

    @classmethod
    def from_path_and_cubepair(cls, path, cubepair, bin_cl = True, forward_model = False, redshift_index = 2.3, include_sn = False, simulate_weighted_mean_subt=False):
        with h5py.File(path , 'r') as f:
            b0_cl_model = np.array(f['b0_cl_model'])
            b1_cl_model = np.array(f['b1_cl_model'])
            b2_cl_model = np.array(f['b2_cl_model'])
            z_edges_model = np.array(f['z_edges'])
        if cubepair.mapcube1.map_array.shape[0] != b0_cl_model.shape[-1]:
            b0_cl_model = b0_cl_model[:, :cubepair.mapcube1.map_array.shape[0], :cubepair.mapcube1.map_array.shape[0]]
            b1_cl_model = b1_cl_model[:, :cubepair.mapcube1.map_array.shape[0], :cubepair.mapcube1.map_array.shape[0]]
            b2_cl_model = b2_cl_model[:, :cubepair.mapcube1.map_array.shape[0], :cubepair.mapcube1.map_array.shape[0]]
            z_edges_model = z_edges_model[:cubepair.mapcube1.map_array.shape[0]]
        ans = cls(b2_cl_model, b1_cl_model, b0_cl_model, cubepair, bin_cl = bin_cl, forward_model = forward_model, redshift_index = redshift_index, include_sn = include_sn, simulate_weighted_mean_subt=simulate_weighted_mean_subt)
        if not np.array_equal(ans.z_edges, z_edges_model):
            print("Warning, model and data have inconsistent redshifts.")
            print(z_edges_model)
            print((ans.z_edges))
        return ans

    def precompute(self, make_amps_meaningful=True, normalize_to_z_index = -1, fix_line_error = True):
        #Precompute effective star-forming mass per redshift slice.
        self.sigmas = int_sigmas(self.redshifts)
        #Precompute redshift dependence of infra-red luminosity.
        self.phi = (1.+ self.redshifts)**self.redshift_index
        #Precompute Hubble parameter.
        self.H = np.sqrt(0.3*(1.+self.redshifts)**3+0.7)
        #Choose not to correct old normalization issue with continuum emission matrix.
        self.fix_continuum_normalization = False
        #If make_amps_meaningful==True, normalize so that line amplitude is the mean line brightness temp and continuum amplitude is the mean continuum brightness temp.
        if make_amps_meaningful:
            if normalize_to_z_index == -1:
                self.cont_norm = np.mean(self.c*self.l_0*self.phi*self.sigmas*self.dz/(4*np.pi*self.H*(1+self.redshifts)))
                if not fix_line_error:
                    self.line_norm = np.mean(self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)))
                else:
                    self.line_norm = np.mean(self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H))
            else:
                self.cont_norm = (self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)))[normalize_to_z_index]
                if not fix_line_error:
                    self.line_norm = (self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)))[normalize_to_z_index]
                else:
                    self.line_norm = (self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H))[normalize_to_z_index]
                #Choose to correct old normalization issue with continuum emission matrix.
                self.fix_continuum_normalization = False
        else:
            self.cont_norm = 1.
            self.line_norm = 1.

    def convolve_freq(self, a_vector, edge_comp = True, normalize = True, norm_to_highest = False):
        if a_vector.size < self.b2_cl_model.shape[-1]:
            a_vector = np.pad(a_vector, (0, self.b2_cl_model.shape[-1] - a_vector.size), 'constant')
        else:
            a_vector = a_vector[:self.b2_cl_model.shape[-1]]
        self.conv_freq = True
        self.a_vector = a_vector
        if edge_comp:
            temp_ind = int(len(a_vector)/2)
            self.a_vector[temp_ind:] = np.zeros(self.a_vector[temp_ind:].shape)
        if normalize:
            a_vector /= np.sum(a_vector) + np.sum(a_vector[1:])
        if norm_to_highest:
            a_vector /= np.max(a_vector)
        z_mat = np.arange(self.b2_cl_model.shape[-1])
        z_mat = np.abs(z_mat[:,None] - z_mat[None,:])
        self.conv_matrix = a_vector[z_mat]
        if edge_comp:
            for ind in range(self.conv_matrix.shape[0]):
                if ind<len(a_vector)/2:
                    self.conv_matrix[ind,2*ind+1:]*=2.
                elif ind>len(a_vector)/2:
                    self.conv_matrix[ind,:2*ind-self.conv_matrix.shape[1]+1]*=2.

    #def dI_dz_cont(self, params, beta=1.5, matrix=True, use_conv = False):
    def dI_dz_cont(self, params, beta=1.5, matrix=True):
        #parms[0] is the amplitude of the continuum spectrum at 1500 GHz.
        #params[1] is the temperature determining the shape of the continuum spectrum.
        if matrix:
            fz_in = self.freqs[:,None]*(self.redshifts[None,:]+1.)
        else:
            fz_in = self.freqs*(self.redshifts+1.)
        if params[0] !=0:
            theta = spect(fz_in, params[0], params[1], beta=beta, fix_normalization = self.fix_continuum_normalization)
        else:
            theta = np.zeros(fz_in.shape)
        ans = (theta*self.c*self.l_0*self.phi*self.sigmas*self.dz/(4*np.pi*self.H*(1+self.redshifts)))/self.cont_norm
        #if self.conv_freq and use_conv:
        #    ans = np.einsum('ij,jk', ans, self.conv_matrix)
        return ans

    def dI_dz_line(self, line_factor, matrix=False, fix_line_error = True):
        #line_factor is the line amplitude.
        if not fix_line_error:
            ans = (line_factor*self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H*(1+self.redshifts)))/self.line_norm
        else:
            ans = (line_factor*self.c*self.l_0*self.phi*self.sigmas/(4*np.pi*self.H))/self.line_norm
        if matrix:
            ans = np.diag(ans)
            #if self.conv_freq:
            #    ans = np.einsum('ij,jk', ans, self.conv_matrix)
        return ans

    def map_output(self, params, b1_map, b0_map=None, beta=1.5):
        #If b0_map is none, then b1_map includes bias and rsd component, with bias factor divided (so RSD part has 1/bias factor).
        if type(b0_map) == type(None):
            b0_map = np.zeros(b1_map.shape)
        cont_func = self.dI_dz_cont(params[0:2], beta = beta, matrix=True)
        cont_map = np.einsum('ij, jk->ik', cont_func, b1_map + b0_map/params[4])
        line_dI_dz = self.dI_dz_line(params[2], matrix=self.conv_freq)
        if not self.conv_freq:
            #line_map = line_dI_dz[:,None]*b2_map
            line_map = line_dI_dz[:,None]*b0_map/params[4]
            #line_map += line_dI_dz[:,None]*b1_map/2.
            line_map += line_dI_dz[:,None]*b1_map
        else:
            #line_map = np.einsum('ij,jk->ki', line_dI_dz, b2_map + b0_map/params[4] + b1_map/2.)
            line_map = np.einsum('ij,jk->ik', line_dI_dz, b0_map/params[4] + b1_map)
        ans = cont_map + line_map
        if self.conv_freq:
            ans = np.einsum('ij,jl->il', self.conv_matrix, ans)
        return ans 

    def cl(self, params, cl_range, beta=1.5):
        #params[0] is the amplitude of the continuum spectrum at 1500 GHz (times the CII/CIB bias).
        #params[1] is the temperature determining the shape of the continuum spectrum.
        #params[2] is the line amplitude (times the CII/CIB bias).
        #params[3] is the galaxy bias.
        #params[4] is the CII/CIB bias.
        #ToDo params[5] for cross shot noise.
        cont_func = self.dI_dz_cont(params[0:2], beta = beta, matrix=True)
        cont_cross_gal = np.einsum('ij,kjl->kil', cont_func, self.b2_cl_model + self.b0_cl_model/(params[3]*params[4]) + self.b1_cl_model/(2.*params[3]) + self.b1_cl_model/(2.*params[4]))
        line_dI_dz = self.dI_dz_line(params[2], matrix=self.conv_freq)
        if not self.conv_freq:
            line_cross_gal = line_dI_dz[None,:,None]*self.b2_cl_model
            line_cross_gal += line_dI_dz[None,:,None]*self.b0_cl_model/(params[3]*params[4])
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[3])
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[4])
        else:
            line_cross_gal = np.einsum('ij,kjl->kil', line_dI_dz, self.b2_cl_model + self.b0_cl_model/(params[3]*params[4]) + self.b1_cl_model/(2.*params[3]) + self.b1_cl_model/(2.*params[4]))
        tot_cross = params[3]*(line_cross_gal + cont_cross_gal)
        if self.conv_freq:
            tot_cross = np.einsum('jk, ikl -> ijl', self.conv_matrix, tot_cross)
        if not self.bin_cl:
            #Apply window functions.
            #Currently assuming one dimensional (ell only, no z-dependence) window functions.
            if type(self.cubepair.mapcube1.win_func) != type(None):
                tot_cross *= self.cubepair.mapcube1.win_func[:,None,None]
            if type(self.cubepair.mapcube2.win_func) != type(None):
                tot_cross *= self.cubepair.mapcube2.win_func[:,None,None]
        return tot_cross[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]

    def d_cl_d_params(self, params, cl_range, beta=1.5):
        #For now, only take derivative w.r.t. continuum amplitude (row 0 of output) and line amplitude (row 1 of output).
        #params[0] is the amplitude of the continuum spectrum at 1500 GHz (times the CII/CIB bias).
        #params[1] is the temperature determining the shape of the continuum spectrum.
        #params[2] is the line amplitude (times the CII/CIB bias).
        #params[3] is the galaxy bias.
        #params[4] is the CII/CIB bias.
        output = np.zeros((2, self.b2_cl_model.shape[0], self.b2_cl_model.shape[1], self.b2_cl_model.shape[2]))
        
        #Continuum derivative first.
        cont_func = self.dI_dz_cont([1., params[1]], beta = beta, matrix=True)
        cont_cross_gal = np.einsum('ij,kjl->kil', cont_func, self.b2_cl_model + self.b0_cl_model/(params[3]*params[4]) + self.b1_cl_model/(2.*params[3]) + self.b1_cl_model/(2.*params[4]))
        line_dI_dz = self.dI_dz_line(1., matrix=self.conv_freq)
        if not self.conv_freq:
            line_cross_gal = line_dI_dz[None,:,None]*self.b2_cl_model
            line_cross_gal += line_dI_dz[None,:,None]*self.b0_cl_model/(params[3]*params[4])
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[3])
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[4])
        else:
            line_cross_gal = np.einsum('ij,kjl->kil', line_dI_dz, self.b2_cl_model + self.b0_cl_model/(params[3]*params[4]) + self.b1_cl_model/(2.*params[3]) + self.b1_cl_model/(2.*params[4]))
        output[0,:] = params[3]*cont_cross_gal
        output[1,:] = params[3]*line_cross_gal
        #tot_cross = params[3]*(line_cross_gal + cont_cross_gal)
        if self.conv_freq:
            output = np.einsum('jk, aikl -> aijl', self.conv_matrix, output)
        if not self.bin_cl:
            #Apply window functions.
            #Currently assuming one dimensional (ell only, no z-dependence) window functions.
            if type(self.cubepair.mapcube1.win_func) != type(None):
                output *= self.cubepair.mapcube1.win_func[None,:,None,None]
            if type(self.cubepair.mapcube2.win_func) != type(None):
                output *= self.cubepair.mapcube2.win_func[None,:,None,None]
        return output

    def d_cl_d_params_natural(self, params, cl_range, beta=1.5):
        #For now, only take derivative w.r.t. continuum amplitude (row 0 of output), line amplitude (row 1 of output), galaxy bias (row 2 of output), and line bias (row 3 of output).
        #params[0] is the amplitude of the continuum spectrum at 1500 GHz.
        #params[1] is the temperature determining the shape of the continuum spectrum.
        #params[2] is the line amplitude.
        #params[3] is the galaxy bias.
        #params[4] is the CII/CIB bias.
        output = np.zeros((4, self.b2_cl_model.shape[0], self.b2_cl_model.shape[1], self.b2_cl_model.shape[2]))

        #Continuum derivative first.
        cont_func = self.dI_dz_cont([params[4], params[1]], beta = beta, matrix=True)
        cont_cross_gal = np.einsum('ij,kjl->kil', cont_func, self.b2_cl_model + self.b0_cl_model/(params[3]*params[4]) + self.b1_cl_model/(2.*params[3]) + self.b1_cl_model/(2.*params[4]))
        line_dI_dz = self.dI_dz_line(params[4], matrix=self.conv_freq)
        if not self.conv_freq:
            line_cross_gal = line_dI_dz[None,:,None]*self.b2_cl_model
            line_cross_gal += line_dI_dz[None,:,None]*self.b0_cl_model/(params[3]*params[4])
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[3])
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[4])
        else:
            line_cross_gal = np.einsum('ij,kjl->kil', line_dI_dz, self.b2_cl_model + self.b0_cl_model/(params[3]*params[4]) + self.b1_cl_model/(2.*params[3]) + self.b1_cl_model/(2.*params[4]))
        output[0,:] = params[3]*cont_cross_gal
        output[1,:] = params[3]*line_cross_gal
        #output[2,:] = self.cl([params[0]*params[4], params[1], params[2]*params[3], 1., params[4]], cl_range)
        cont_func = self.dI_dz_cont([params[0], params[1]], beta = beta, matrix=True)
        cont_cross_gal = np.einsum('ij,kjl->kil', cont_func, self.b2_cl_model + self.b1_cl_model/(2.*params[4]))
        line_dI_dz = self.dI_dz_line(params[2], matrix=self.conv_freq)
        if not self.conv_freq:
            line_cross_gal = line_dI_dz[None,:,None]*self.b2_cl_model
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[4])
        else:
            line_cross_gal = np.einsum('ij,kjl->kil', line_dI_dz, self.b2_cl_model + self.b1_cl_model/(2.*params[4]))
        output[2,:] = params[4]*cont_cross_gal + params[4]*line_cross_gal

        cont_func = self.dI_dz_cont([params[0], params[1]], beta = beta, matrix=True)
        cont_cross_gal = np.einsum('ij,kjl->kil', cont_func, self.b2_cl_model + self.b1_cl_model/(2.*params[3]))
        line_dI_dz = self.dI_dz_line(params[2], matrix=self.conv_freq)
        if not self.conv_freq:
            line_cross_gal = line_dI_dz[None,:,None]*self.b2_cl_model
            line_cross_gal += line_dI_dz[None,:,None]*self.b1_cl_model/(2.*params[3])
        else:
            line_cross_gal = np.einsum('ij,kjl->kil', line_dI_dz, self.b2_cl_model + self.b1_cl_model/(2.*params[3]))
        output[3,:] = params[3]*cont_cross_gal + params[3]*line_cross_gal 
        #tot_cross = params[3]*(line_cross_gal + cont_cross_gal)
        if self.conv_freq:
            output = np.einsum('jk, aikl -> aijl', self.conv_matrix, output)
        if not self.bin_cl:
            #Apply window functions.
            #Currently assuming one dimensional (ell only, no z-dependence) window functions.
            if type(self.cubepair.mapcube1.win_func) != type(None):
                output *= self.cubepair.mapcube1.win_func[None,:,None,None]
            if type(self.cubepair.mapcube2.win_func) != type(None):
                output *= self.cubepair.mapcube2.win_func[None,:,None,None]
        return output[:,cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]

    def calc_auto_no_foregrounds(self, params, cl_range, beta=1.5):
        #Calculates what the microwave auto-power would be without foregrounds (so just due to CII and CIB).
        #params[0] is the amplitude of the continuum spectrum at 1500 GHz (times the CII/CIB bias).
        #parms[1] is the temperature determining the shape of the continuum spectrum.
        #params[2] is the line amplitude (times the CII/CIB bias).
        #params[3] is the CII/CIB bias.
        cont_spect = self.dI_dz_cont(params[0:2], beta = beta, matrix=True)
        line_spect = self.dI_dz_line(params[2], matrix=True)
        tot_spect = cont_spect + line_spect
        #tot_spect_uncov = copy.deepcopy(tot_spect)
        if self.conv_freq:
            tot_spect = np.dot(self.conv_matrix, tot_spect)
        ans = np.einsum('jk,ikl,ml->ijm', tot_spect, self.b2_cl_model, tot_spect)
        ans += np.einsum('jk,ikl,ml->ijm', tot_spect, self.b0_cl_model, tot_spect)/(params[3]**2.)
        ans += np.einsum('jk,ikl,ml->ijm', tot_spect, self.b1_cl_model, tot_spect)/params[3]
        ans = ans[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        #return [tot_spect, ans, tot_spect_uncov]
        return ans



class cl_auto_and_cross():
#Cl model for full auto and cross.
#Format explained below, using BOSS and FIRAS as an example:
#Output Cl(ell, ind, ind'): ind has 2z elements, where z is the num. of redshift slices. The first z are BOSS and the last z are FIRAS.
#Upper right corner is thus BOSSxFIRAS, and lower left corner is FIRASxBOSS.
#Upper left corner is BOSSxBOSS, and lower right corner is FIRASxFIRAS.

#This will forward model automatically, using mixing_matrix and beam_functions and bin_params.
#To avoid forward modeling, window_func1 (microwave) and window_func2 (gal) should be all ones, mixing matrix should be identity.
#For forward modeling, cl_model and cl_model_larger should both be unbinned: cl_model_larger can extend to farther redshifts to capture continuum correlations outside survey redshifts.

    cl_model = None
    cl_model_larger = None
    freqs = None
    redshifts = None
    #The redshift_index is the power law index for the (1+z)^power dependence of the luminosity.
    redshift_index = None
    z_indeces_to_keep = None
    sn_model = None
    #Array of consecutive ell-vectors to be considered.
    ell_vect = None
    #Pixel and beam window function for microwave data. Dimensions: (z,ell). If data is beam corrected, use all ones.
    window_func1 = None
    #Pixel and beam window function for galaxy survey data. Dimensions: (z,ell). If data is beam corrected, use all ones.
    window_func2 = None
    #The ell-ell' mixing matrices for the auto-gal, cross, and auto-microwave: dimension is (3, ell,ell')
    #Index 0 should be galxgal mixing, index 1 should be microwavexgal mixing, and index 2 should be microwavexmicrowave mixing.
    #This assumes all galaxy maps have the same weights and all microwave maps have the same weights, possibly different from galaxy map weights.
    #If data is not forward modeled, then for each index, set mixing_matrix[ind,:,:] = identity.
    mixing_matrix = None
    #Length 2 array: bin_params[0] is the ell_vect index to start binning, and bin_params[1] is the bin size.
    bin_params = None
    #Thermal noise model, cl(ell,z,z').
    thermal_noise = None

    #No CIB and CII term in auto-power by default. If it is desired, set this parmeter to True.
    signal_in_auto = False

    def __init__(self, cl_model=None, cl_model_larger=None, freqs=None, redshifts=None, redshift_index=None, z_indeces_to_keep=None, sn_model=None, ell_vect=None, window_func1=None, window_func2=None, mixing_matrix=None, bin_params=None, thermal_noise=None, dummy=False):
        self.cl_model = cl_model
        if type(cl_model_larger) != type(None):
            self.cl_model_larger = cl_model_larger
        else:
            self.cl_model_larger = copy.deepcopy(cl_model)
        #Forward model self.cl_model if ells are unbinned. This will be used for the galaxy auto-power.
        if self.cl_model.shape[0] == mixing_matrix.shape[-1]:
            self.cl_model = forward_model(self.cl_model, mixing_matrix[0,:], window_func2, window_func2, bin_params, ell_vect)
        self.freqs = freqs
        self.redshifts = redshifts
        self.redshift_index = redshift_index
        if type(z_indeces_to_keep) != type(None):
            self.z_indeces_to_keep = z_indeces_to_keep
        else:
            self.z_indeces_to_keep = [0,redshifts.size]
        #Forward model self.sn_model if ells are unbinned.
        if sn_model.shape[0] == mixing_matrix.shape[-1]:
            self.sn_model = forward_model(sn_model, mixing_matrix[0,:], window_func2, window_func2, bin_params, ell_vect)
        else:
            self.sn_model = sn_model
        self.ell_vect = ell_vect
        self.window_func1 = window_func1
        self.window_func2 = window_func2
        self.mixing_matrix = mixing_matrix
        self.bin_params = bin_params
        self.thermal_noise = thermal_noise
        self.dummy = dummy

    def precompute(self):
        self.gal_auto = gal_auto(self.cl_model, self.sn_model)
        #Noise is assumed to be premixed and binned if the number of ell bins is not equal to the mixing matrix size.
        noise_bool = self.thermal_noise.shape[0] != self.mixing_matrix.shape[-1]
        self.milky_way_auto = milky_way_auto(self.ell_vect, self.window_func1, self.mixing_matrix[2,:], self.bin_params, self.freqs, self.thermal_noise, noise_premixed = noise_bool)
        self.cross_power = cross_power(self.cl_model_larger, self.redshifts, self.freqs, self.redshift_index, z_indeces_to_keep = self.z_indeces_to_keep, mixing_matrix = self.mixing_matrix[1,:], window1=self.window_func1, window2=self.window_func2, bin_params=self.bin_params)
        self.cross_power.precompute()

    def cl(self, params, cl_range):
        #params[0] is galaxy bias.
        #params[1] is galaxy shot noise multiplier.
        #params[2] is Galaxtic emission power spectrum index in ell.
        #params[3] is the amplitude multiplying the Galactic emission SED (at ell=1).
        #params[4] is the temperature of the grey-body model for Galactic emission.
        #params[5] is the thermal_noise amplitude multiplier.
        #params[6] is the amplitude of the continuum spectrum at 1500 GHz.
        #params[7] is the temperature determining the shape of the CIB continuum spectrum.
        #params[8] is the line amplitude.
        #If dummy is true then params[9] through params[11] are the CIB, amplitude, temperature, and CII line amplitude in the auto-power.
 
        cl_gal = self.gal_auto.cl(params[0:2], cl_range)
        ell_size = cl_range[1] - cl_range[0]
        z_size = (cl_range[3] - cl_range[2])/2
        cl_mw = self.milky_way_auto.cl(params[2:6], cl_range)
        cl_cross = self.cross_power.cl(np.concatenate((params[6:9], [params[0]])), cl_range)
        full_cl = np.zeros((ell_size, 2*z_size, 2*z_size))
        full_cl[:,:z_size,:z_size] = cl_gal
        full_cl[:,z_size:,z_size:] = cl_mw
        if self.signal_in_auto == True:
            if not self.dummy:
                full_cl[:,z_size:, z_size:] += self.cross_power.calc_auto_no_foregrounds(params[6:9], cl_range)
            else:
                full_cl[:,z_size:, z_size:] += self.cross_power.calc_auto_no_foregrounds(params[9:12], cl_range)
                full_cl[:,z_size:, z_size:] += self.cross_power.calc_auto_no_foregrounds(params[6:9], cl_range) 
        full_cl[:,z_size:,:z_size] = cl_cross
        full_cl[:,:z_size, z_size:] = np.einsum('ijk->ikj', cl_cross)
        return full_cl

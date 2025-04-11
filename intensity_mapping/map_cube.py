import numpy as np
import scipy as sp
import healpy as hp
import h5py
import copy
from itertools import product
import pymaster as nmt

class MapCube(object):

    def __init__(self, map_array, weights, z_edges = None, win_func = None):
        if map_array.ndim != 1 and map_array.ndim != 2:
            raise TypeError("Map array must have dimension 1 or 2.")
        nside = (map_array.shape[-1]/12)**0.5
        if nside != int(nside):
            raise TypeError("Maps must be in HealPix format. Trailing dimension must have size 12*nside^2.")
        else:
            self.nside = int(nside)
        self.map_array = map_array
        if weights.ndim != 1 and weights.ndim != 2:
            raise TypeError("Weights must have dimension 1 or 2.")
        if (weights.shape[-1]/12)**0.5 != self.nside:
            raise TypeError("Weights must match nside of maps.")
        self.weights = weights
        self.z_edges = z_edges
        if type(win_func) == type(None):
            self.win_func = None
        elif win_func.ndim != 1 and win_func.ndim != 2:
            raise TypeError("Window function must have dimension 1 or 2.")
        elif win_func.shape[-1] != 3*self.nside:
            raise TypeError("Window function must have trailing dimension 3*nside.")
        else:
            self.win_func = win_func
        self.templates = None

    @classmethod
    def from_gal_counts(cls, gal_count_array, raw_sel_func, z_edges = None, win_func = None, make_sel_func_separable = True, cut_sel_func_extremes = True, theta_range = [0, np.pi/2], cut_factor = 1.308765):
        #gal_count_array is the real galaxy number count.
        #raw_sel_func is the expected unclustered galaxy count.
        init_obj = cls(gal_count_array, raw_sel_func, z_edges, win_func)
        theta_data = hp.pix2ang(init_obj.nside, np.arange(12*init_obj.nside**2))[0]
        cut_bool = np.logical_or(theta_data<theta_range[0], theta_data>theta_range[1])*np.ones(gal_count_array.shape, dtype=bool)
        gal_count_array[cut_bool] = 0.
        raw_sel_func[cut_bool] = 0.
        if make_sel_func_separable:
            z_func = np.sum(raw_sel_func, axis=1)
            theta_func = np.sum(raw_sel_func, axis=0)
            sep_sel = z_func[:,None]*theta_func[None,:]
            sep_sel *= np.sum(gal_count_array)/np.sum(sep_sel)
            if cut_sel_func_extremes:
                #Cut values more than cut_factor standard deviations below mean.
                mean = np.mean(theta_func)
                std = np.std(theta_func)
                cut_bool = (theta_func < mean - cut_factor*std)*np.ones(gal_count_array.shape, dtype=bool)
                gal_count_array[cut_bool] = 0.
                sep_sel[cut_bool] = 0.
                raw_sel_func = sep_sel
        non_zero_bool = raw_sel_func != 0
        map_array = np.zeros(gal_count_array.shape)
        map_array[non_zero_bool] = gal_count_array[non_zero_bool]/raw_sel_func[non_zero_bool] - 1.
        return cls(map_array, raw_sel_func, z_edges, win_func)
       
    @classmethod         
    def from_firas_data(cls, freq_range_to_keep = [0, 3000], line="CII", final_nside = 128, window_func_file = None, thermal_window_func_file = None):
        from intensity_mapping import firas_im as fir
        firas_data = fir.read_firas()
        edges = np.logical_and(firas_data[1]>=np.min(freq_range_to_keep), firas_data[1]<=np.max(freq_range_to_keep))
        z_size = np.sum(edges)
        map_edges = edges[None,:]*np.ones(firas_data[0].shape, dtype=np.bool)
        map_array = (firas_data[0])[map_edges].reshape((12*16**2, z_size)).T
        weights = (firas_data[3][edges]**-2)[:,None]*firas_data[2][None,:]
        zs = fir.define_z(firas_data[1][edges], line=line)
        z_edges = np.concatenate((zs[1][zs[2]][:-1],zs[0][zs[2]][-2:]), axis=0)
        ans = cls(map_array, weights, z_edges = z_edges)
        ans.frequencies = firas_data[1][edges]
        if final_nside != 16:
            ans.change_nside(final_nside, weight_type = 'thermal')
        if type(window_func_file) != type(None):
            with h5py.File(window_func_file, 'r') as f:
                win_func = np.ones((3*final_nside))
                win_func[2:3*final_nside] = np.array(f['full_FIRAS_128'])**0.5
                ans.win_func = win_func
        else:
            ans.win_func = np.ones((3*final_nside))
        if type(thermal_window_func_file) != type(None):
            with h5py.File(thermal_window_func_file, 'r') as f:
                thermal_win_func = np.ones((3*final_nside))
                thermal_win_func[2:3*final_nside] = np.array(f['pixel_transfer'])**0.5
                #thermal_win_func[2:3*final_nside] = np.array(f['pixel_transfer'])**-0.5
                ans.thermal_win_func = thermal_win_func
        else:
            ans.thermal_win_func = np.ones((3*final_nside))
        return ans

    def change_nside(self, new_nside, update_win_func = True, weight_type = 'gal_selection'):
        if weight_type == 'gal_selection':
            #Keep sum of selection function the same.
            power_number = -2
        elif weight_type == 'thermal':
            #Keep noise variance times omega_pixel the same.
            power_number = 2
            self.weights = self.weights**-1.
        else:
            #Keep mean of noise the same.
            power_number = 0 
        if type(self.win_func) != type(None):
            if update_win_func:
                self.win_func = self.win_func[:3*new_nside]*hp.sphtfunc.pixwin(new_nside)
        if self.map_array.ndim == 2:
            new_map_array = np.zeros((self.map_array.shape[0], 12*new_nside**2))
            for z in range(self.map_array.shape[0]):
                new_map_array[z,:] = hp.ud_grade(self.map_array[z,:], new_nside)
        else:
            new_map_array = hp.ud_grade(self.map_array, new_nside)
        if self.weights.ndim == 2:
            new_weights = np.zeros((self.weights.shape[0], 12*new_nside**2))
            for z in range(self.weights.shape[0]):
                #if maintain_weight_sum:
                #    new_weights[z,:] = hp.ud_grade(self.weights[z,:], new_nside, power=-2)
                #else:
                #    new_weights[z,:] = hp.ud_grade(self.weights[z,:], new_nside)
                new_weights[z,:] = hp.ud_grade(self.weights[z,:], new_nside, power=power_number)
        else:
            #if maintain_weight_sum:
            #    new_weights = hp.ud_grade(self.weights, new_nside, power=-2)
            #else:
            #    new_weights = hp.ud_grade(self.weights, new_nside)
            new_weights = hp.ud_grade(self.weights, new_nside, power=power_number)
        self.nside = new_nside
        self.map_array = new_map_array
        self.weights = new_weights
        if weight_type == 'thermal':
            self.weights = self.weights**-1.

    @classmethod
    def from_saved(cls, path):
        with h5py.File(path , 'r') as f:
            map_array = np.array(f['map_array'])
            weights = np.array(f['weights'])
            try:
                z_edges = np.array(f['z_edges'])
            except KeyError:
                z_edges = None
            try:
                win_func = np.array(f['win_func'])
            except KeyError:
                win_func = None
            return cls(map_array, weights, z_edges, win_func)

    @classmethod
    def from_dict(cls, f):
        map_array = np.array(f['map_array'])
        weights = np.array(f['weights'])
        try:
            z_edges = np.array(f['z_edges'])
        except KeyError:
            z_edges = None
        try:
            win_func = np.array(f['win_func'])
        except KeyError:
            win_func = None
        return cls(map_array, weights, z_edges, win_func)

    def save(self, path):
        with h5py.File(path, 'w') as f:
            for key in list(self.__dict__.keys()):
                if type(self.__dict__[key]) == np.ndarray:
                    f[key] = self.__dict__[key]

    def make_map_dict(self, subtract_weighted_mean = True, separable_weights = True, use_wmean_deprojection = False):
        import pymaster as nmt
        self.map_dict = {}
        self.map_dict_no_win_func = {}
        self.separable_weights = separable_weights
        if self.map_array.ndim == 2:
            num_z = self.map_array.shape[0]
        else:
            num_z = 1
        for z in range(num_z):
            #if num_z>1:
            if self.map_array.ndim == 2:
                this_map = self.map_array[z,:]
            else:
                this_map = self.map_array
            if separable_weights and self.weights.ndim == 2:
                this_weight = self.weights[0,:]
            elif self.weights.ndim == 1:
                this_weight = self.weights
            else:
                this_weight = self.weights[z,:]
            if type(self.win_func) == type(None):
                this_win_func = self.win_func
            elif self.win_func.ndim == 2:
                this_win_func = self.win_func[z,:]
            else:
                this_win_func = self.win_func
            this_map = copy.deepcopy(this_map)
            if subtract_weighted_mean:
                if not use_wmean_deprojection:
                    this_map -= np.sum(this_weight*this_map)/np.sum(this_weight)
                else:
                    self.templates = [[np.ones(this_map.shape)]]
            self.map_dict[str(z)]=nmt.NmtField(this_weight, [this_map], templates = self.templates, beam = this_win_func)
            self.map_dict_no_win_func[str(z)]=nmt.NmtField(this_weight, [this_map], templates = self.templates, beam = None)

    def compute_coupling_matrix(self, ell_bin_size = 9):
        import pymaster as nmt
        print('Bin size set to ' + str(ell_bin_size) + ' for nside ' + str(self.nside) + '.')
        self.ell_bin_size = ell_bin_size
        self.bins = nmt.NmtBin(self.nside, nlb=ell_bin_size)
        if self.separable_weights:
            self.coupling_matrix = nmt.NmtWorkspace()
            self.coupling_matrix.compute_coupling_matrix(self.map_dict['0'], self.map_dict['0'], self.bins)
        else:
            self.coupling_matrix = {}
            for key in product(list(self.map_dict.keys()), list(self.map_dict.keys())):
                self.coupling_matrix[key[0]+key[1]] = nmt.NmtWorkspace()
                self.coupling_matrix[key[0]+key[1]].compute_coupling_matrix(self.map_dict[key[0]], self.map_dict[key[1]], self.bins)

    def compute_ell_coupling(self):
        import pymaster as nmt
        map_dict = self.map_dict
        if self.separable_weights:
            self.ell_coupling = nmt.NmtCovarianceWorkspace()
            self.ell_coupling.compute_coupling_coefficients(map_dict['0'], map_dict['0'], map_dict['0'], map_dict['0'])
            #self.ell_coupling.compute_coupling_coefficients(self.coupling_matrix,self.coupling_matrix)
        else:
            self.ell_coupling = {}
            for key in product(list(self.coupling_matrix.keys()), list(self.coupling_matrix.keys())):
                for key2 in product(list(self.coupling_matrix.keys()), list(self.coupling_matrix.keys())):
                    self.ell_coupling[key[0]+key[1]+key2[0]+key2[1]] = nmt.NmtCovarianceWorkspace()
                    #self.ell_coupling[key[0]+key[1]].compute_coupling_coefficients(self.coupling_matrix[key[0]], self.coupling_matrix[key[1]])
                    self.ell_coupling[key[0]+key[1]+key2[0]+key2[1]].compute_coupling_matrix(map_dict[key[0]], map_dict[key[1]], map_dict[key2[0]], map_dict[key2[1]])

    def compute_cl(self, forward_model = False):
        import pymaster as nmt
        num_z = len(list(self.map_dict.keys()))
        self.ells = np.array(self.bins.get_effective_ells())
        if not forward_model:
            self.cl_zz = np.zeros((self.ells.size, num_z, num_z))
        else:
            from . import pwrspec_estimation as pe
            ell_vec = np.arange(3*self.nside)
            self.pseudo_cl_zz = np.zeros((self.ells.size, num_z, num_z))
        for key in product(list(self.map_dict.keys()), list(self.map_dict.keys())):
           c_coupled=nmt.compute_coupled_cell(self.map_dict[key[0]],self.map_dict[key[1]])
           if self.separable_weights:
               coupling_mat = self.coupling_matrix
           else:
               coupling_mat = self.coupling_matrix[key[0]+key[1]]
           if not forward_model:
               self.cl_zz[:,int(key[0]),int(key[1])]=coupling_mat.decouple_cell(c_coupled)
           else:
               self.pseudo_cl_zz[:,int(key[0]),int(key[1])] = pe.bin_power(ell_vec[2:], c_coupled[0,2:] ,self.ell_bin_size, ell_axis = 0)[1]

    def bin_cl_model(self, cl_model, forward_model = False, simulate_weighted_mean_subt = False):
        import pymaster as nmt
        self.ells = np.array(self.bins.get_effective_ells())
        binned_cl_model = np.zeros((self.ells.size, cl_model.shape[1], cl_model.shape[2]))
        for key in product(list(self.map_dict.keys()), list(self.map_dict.keys())):
           if self.separable_weights:
               coupling_mat = self.coupling_matrix
           else:
               coupling_mat = self.coupling_matrix[key[0]+key[1]]
           c_coupled = coupling_mat.couple_cell([cl_model[:, int(key[0]), int(key[1])]])
           if not forward_model:
               binned_cl_model[:,int(key[0]),int(key[1])]=coupling_mat.decouple_cell(c_coupled)
           else:
               from . import pwrspec_estimation as pe
               ell_vec = np.arange(3*self.nside)
               binned_cl_model[:,int(key[0]),int(key[1])] = pe.bin_power(ell_vec[2:], c_coupled[0,2:] ,self.ell_bin_size, ell_axis = 0)[1]
        return binned_cl_model

    def full_compute(self, subtract_weighted_mean = True, separable_weights = True, ell_bin_size = 9, forward_model = False, use_wmean_deprojection = False):
        print('Creating dictionary of NmtField objects for each redshift.')
        self.make_map_dict(subtract_weighted_mean = subtract_weighted_mean, separable_weights = separable_weights, use_wmean_deprojection = use_wmean_deprojection)
        print('Calculating coupling matrix for NmtWorkspace.')
        self.compute_coupling_matrix(ell_bin_size = ell_bin_size)
        print('Computing Cl_zz matrix.')
        self.compute_cl(forward_model = forward_model)
        print('Computing ell coupling for Cl covariance.')
        self.compute_ell_coupling()

    def compute_unbinned_mixing_matrix(self):
        if self.separable_weights:
            from . import sphere_music as sm
            self.unbinned_mixing_matrix = sm.mixing_from_weight(self.weights[0,:], self.weights[0,:])

    def compute_binning(self, startl=2):
        from . import pwrspec_estimation as pe
        ell_vec = np.arange(3*self.nside)
        bins = pe.make_regular_bins(ell_vec, self.ell_bin_size, startl=startl, fill_irregular_ends = False)
        ell_bins, self.binning, self.unbinning = pe.bin_unbin(ell_vec, bins)

    def compute_binned_mixing_matrix(self, startl=2):
        try:
            unbinned_mixing_matrix = self.unbinned_mixing_matrix
        except AttributeError:
            self.compute_unbinned_mixing_matrix()
        try:
            binning = self.binning
        except AttributeError:
            self.compute_binning(startl=startl)
        self.binned_mixing_matrix = np.einsum('ij,jk,kl', self.binning, self.unbinned_mixing_matrix, self.unbinning)

    def compute_pseudo_cl_cov_from_cl_cov(self, cl_cov, startl=2):
        if self.separable_weights:
            ell_size = np.shape(cl_cov)[0] 
            try:
                binned_mixing_matrix = self.binned_mixing_matrix
            except AttributeError:
                self.compute_binned_mixing_matrix(startl=startl)
            return np.einsum('ij, jklm, ln->iknm', self.binned_mixing_matrix[:ell_size,:ell_size], cl_cov, self.binned_mixing_matrix[:ell_size,:ell_size].T)

class CubePair(object):

    def __init__(self, mapcube1, mapcube2):
        self.mapcube1 = mapcube1
        self.mapcube2 = mapcube2
        if mapcube1.map_array.shape != mapcube2.map_array.shape:
            raise TypeError("Map cubes expected to have same shape.")
        self.nside = mapcube1.nside

    def save(self, path):
        with h5py.File(path, 'w') as f:
            for key in list(self.__dict__.keys()):
                if type(self.__dict__[key]) == np.ndarray:
                    f[key] = self.__dict__[key]
                #if isinstance(self.__dict__[key], MapCube):
                if key == 'mapcube1' or key == 'mapcube2':
                    #f[key] = {}
                    f.create_group(key)
                    for key2 in list(self.__dict__[key].__dict__.keys()):
                        if type(self.__dict__[key].__dict__[key2]) == np.ndarray:
                            f[key][key2] = self.__dict__[key].__dict__[key2]

    @classmethod
    def from_saved(cls, path):
        with h5py.File(path , 'r') as f:
            mapcube1 = MapCube.from_dict(f['mapcube1'])
            mapcube2 = MapCube.from_dict(f['mapcube2'])
            ans = cls(mapcube1, mapcube2)
            for key in list(f.keys()):
                if key != 'mapcube1' and key != 'mapcube2':
                    setattr(ans, key, np.array(f[key]))
        return ans

    def complete_cross_compute(self, subtract_weighted_mean1 = True, separable_weights1 = True, subtract_weighted_mean2 = True, separable_weights2 = True, ell_bin_size = 9, use_wmean_deprojection = False):
        #try:
        #    self.map_dict1 = self.mapcube1.map_dict
        #except AttributeError:
        print('Creating dictionary of NmtField objects for each redshift for first cube.')
        self.mapcube1.make_map_dict(subtract_weighted_mean = subtract_weighted_mean1, separable_weights = separable_weights1, use_wmean_deprojection = use_wmean_deprojection)
        self.map_dict1 = self.mapcube1.map_dict
        #try:
        #    self.map_dict2 = self.mapcube2.map_dict
        #except AttributeError:
        print('Creating dictionary of NmtField objects for each redshift for second cube.')
        self.mapcube2.make_map_dict(subtract_weighted_mean = subtract_weighted_mean2, separable_weights = separable_weights2, use_wmean_deprojection = use_wmean_deprojection)
        self.map_dict2 = self.mapcube2.map_dict
        #try:
        #    cross_coupling_matrix = self.cross_coupling_matrix
        #except AttributeError:
        print('Calculating cross coupling matrix for NmtWorkspace.')
        self.compute_cross_coupling_matrix(ell_bin_size = ell_bin_size)
        cross_coupling_matrix = self.cross_coupling_matrix
        #try:
        #    cl_zz = self.cl_zz
        #except AttributeError:
        print('Computing Cl_zz cross-power matrix.')
        self.compute_cl()
        #try:
        #    ell_coupling = self.ell_coupling
        #except AttributeError:
        print('Computing ell coupling for Cl covariance.')
        self.compute_ell_coupling()

    def compute_cross_coupling_matrix(self, ell_bin_size = 9):
        import pymaster as nmt
        print('Bin size set to ' + str(ell_bin_size) + ' for nside ' + str(self.nside) + '.')
        self.ell_bin_size = ell_bin_size
        self.bins = nmt.NmtBin(self.nside, nlb=ell_bin_size)
        if self.mapcube1.separable_weights and self.mapcube2.separable_weights:
            self.both_weights_separable = True
            self.cross_coupling_matrix = nmt.NmtWorkspace()
            self.cross_coupling_matrix.compute_coupling_matrix(self.map_dict1['0'], self.map_dict2['0'], self.bins)
        else:
            self.both_weights_separable = False
            self.cross_coupling_matrix = {}
            for key in product(list(self.map_dict1.keys()), list(self.map_dict2.keys())):
                self.cross_coupling_matrix[key[0]+key[1]] = nmt.NmtWorkspace()
                self.cross_coupling_matrix[key[0]+key[1]].compute_coupling_matrix(self.map_dict1[key[0]], self.map_dict2[key[1]], self.bins)
        try:
            self.thermal_transfer_func = self.mapcube1.thermal_win_func*self.mapcube2.thermal_win_func
            if self.mapcube1.separable_weights and self.mapcube2.separable_weights:
                self.thermal_cross_coupling_matrix = nmt.NmtWorkspace()
                self.thermal_cross_coupling_matrix.compute_coupling_matrix(self.mapcube1.map_dict_no_win_func['0'], self.mapcube2.map_dict_no_win_func['0'], self.bins)
            else:
                self.both_weights_separable = False
                self.thermal_cross_coupling_matrix = {}
                for key in product(list(self.map_dict1.keys()), list(self.map_dict2.keys())):
                    self.thermal_cross_coupling_matrix[key[0]+key[1]] = nmt.NmtWorkspace()
                    self.thermal_cross_coupling_matrix[key[0]+key[1]].compute_coupling_matrix(self.mapcube1.map_dict_no_win_func[key[0]], self.mapcube2.map_dict_no_win_func[key[1]], self.bins)
        except AttributeError:
            print("Not computing thermal transfer function and coupling matrix.")

    def compute_cl(self, forward_model=False, start_pseudo_ell = 2):
        import pymaster as nmt
        num_z = len(list(self.map_dict1.keys()))
        self.ells = np.array(self.bins.get_effective_ells())
        if not forward_model:
            self.cl_zz = np.zeros((self.ells.size, num_z, num_z))
        else:
            from . import pwrspec_estimation as pe
            ell_vec = np.arange(3*self.nside)
            if start_pseudo_ell == 2:
                self.pseudo_cl_zz = np.zeros((self.ells.size, num_z, num_z))
            else:
                self.pseudo_cl_zz = np.zeros((3*self.mapcube1.nside - start_pseudo_ell, num_z, num_z))
        for key in product(list(self.map_dict1.keys()), list(self.map_dict2.keys())):
           c_coupled=nmt.compute_coupled_cell(self.map_dict1[key[0]],self.map_dict2[key[1]])
           if self.both_weights_separable:
               coupling_mat = self.cross_coupling_matrix
           else:
               coupling_mat = self.cross_coupling_matrix[key[0]+key[1]]
           if not forward_model:
               self.cl_zz[:,int(key[0]),int(key[1])]=coupling_mat.decouple_cell(c_coupled)
           else:
               self.pseudo_cl_zz[:,int(key[0]),int(key[1])] = pe.bin_power(ell_vec[start_pseudo_ell:], c_coupled[0,start_pseudo_ell:] ,self.ell_bin_size, ell_axis = 0)[1]

    def bin_cl_model(self, cl_model, forward_model = False, thermal_noise = False, window_func_preapplied = False, simulate_weighted_mean_subt=False, pcl_correction = None):
        #Weighted mean subtraction simulation assumes weighted mean removed via NaMaster mode deprojection. Currently assumes separable weights.
        import pymaster as nmt
        self.ells = np.array(self.bins.get_effective_ells())
        binned_cl_model = np.zeros((self.ells.size, cl_model.shape[1], cl_model.shape[2]))
        if thermal_noise:
            cl_model = cl_model*self.thermal_transfer_func[:,None,None]
        for key in product(list(self.map_dict1.keys()), list(self.map_dict2.keys())):
           if self.both_weights_separable:
               decoupling_mat = self.cross_coupling_matrix
               if thermal_noise or window_func_preapplied:
                   coupling_mat = self.thermal_cross_coupling_matrix
                   if simulate_weighted_mean_subt:
                       cl_bias = nmt.deprojection_bias(self.mapcube1.map_dict_no_win_func['0'], self.mapcube2.map_dict_no_win_func['0'], [cl_model[:, int(key[0]), int(key[1])]])
               else:
                   coupling_mat = self.cross_coupling_matrix
                   if simulate_weighted_mean_subt:
                       cl_bias = nmt.deprojection_bias(self.mapcube1.map_dict['0'], self.mapcube2.map_dict['0'], [cl_model[:, int(key[0]), int(key[1])]])
           else:
               decoupling_mat = self.cross_coupling_matrix[key[0]+key[1]]
               if thermal_noise or window_func_preapplied:
                   coupling_mat = self.thermal_cross_coupling_matrix[key[0]+key[1]]
                   if simulate_weighted_mean_subt:
                       cl_bias = nmt.deprojection_bias(self.mapcube1.map_dict_no_win_func[key[0]], self.mapcube2.map_dict_no_win_func[key[1]], [cl_model[:, int(key[0]), int(key[1])]])
               else:
                   coupling_mat = self.cross_coupling_matrix[key[0]+key[1]]
                   if simulate_weighted_mean_subt:
                       cl_bias = nmt.deprojection_bias(self.mapcube1.map_dict[key[0]], self.mapcube2.map_dict[key[1]], [cl_model[:, int(key[0]), int(key[1])]])
           c_coupled = coupling_mat.couple_cell([cl_model[:, int(key[0]), int(key[1])]])
           if type(pcl_correction) != type(None):
               c_coupled[0,:] *= pcl_correction
           if simulate_weighted_mean_subt:
               c_coupled[0,:] += cl_bias[0,:]
           if not forward_model:
               binned_cl_model[:,int(key[0]),int(key[1])]=decoupling_mat.decouple_cell(c_coupled)
           else:
               from . import pwrspec_estimation as pe
               ell_vec = np.arange(3*self.nside)
               binned_cl_model[:,int(key[0]),int(key[1])] = pe.bin_power(ell_vec[2:], c_coupled[0,2:] ,self.ell_bin_size, ell_axis = 0)[1]
        return binned_cl_model

    def bin_cl_model_ell_only(self, cl_model, forward_model = False, simulate_weighted_mean_subt=False, pcl_correction = None):
        #Only works if cross coupling matrix is separable.
        #Applies regular coupling (including beam effects), so don't use on thermal noise.
        ans = self.cross_coupling_matrix.couple_cell([cl_model])
        if type(pcl_correction) != type(None):
            ans[0,:] *= pcl_correction
        if simulate_weighted_mean_subt:
            #Assume weighted mean removed via NaMaster mode deprojection.
            #Currently assumes separable weights.
            cl_bias = nmt.deprojection_bias(self.mapcube1.map_dict['0'], self.mapcube2.map_dict['0'], [cl_model])
            ans[0,:] += cl_bias[0,:]
        if not forward_model:
            #print 'not forward modelled.'
            ans = self.cross_coupling_matrix.decouple_cell(ans)
            #print ans.shape
            ans = ans[0,:]
        else:
            #print 'forward modelled.'
            from . import pwrspec_estimation as pe
            ell_vec = np.arange(3*self.nside)
            ans = pe.bin_power(ell_vec[2:], ans[0,2:] ,self.ell_bin_size, ell_axis = 0)[1]
        return ans

    def compute_unbinned_mixing_matrix(self):
        if self.both_weights_separable:
            from . import sphere_music as sm
            if self.mapcube1.weights.ndim == 2:
                self.unbinned_mixing_matrix = sm.mixing_from_weight(self.mapcube1.weights[0,:], self.mapcube2.weights[0,:])
            else:
                self.unbinned_mixing_matrix = sm.mixing_from_weight(self.mapcube1.weights, self.mapcube2.weights)

    def compute_binning(self, startl=2):
        from . import pwrspec_estimation as pe
        ell_vec = np.arange(3*self.nside)
        bins = pe.make_regular_bins(ell_vec, self.ell_bin_size, startl=startl, fill_irregular_ends = False)
        ell_bins, self.binning, self.unbinning = pe.bin_unbin(ell_vec, bins)

    def compute_binned_mixing_matrix(self, startl=2):
        try:
            unbinned_mixing_matrix = self.unbinned_mixing_matrix
        except AttributeError:
            self.compute_unbinned_mixing_matrix()
        try:
            binning = self.binning
        except AttributeError:
            self.compute_binning(startl=startl)
        if type(self.mapcube1.win_func) != type(None) and type(self.mapcube2.win_func) != type(None):
            self.tfunc = self.mapcube1.win_func*self.mapcube2.win_func
        elif type(self.mapcube1.win_func) != type(None) and type(self.mapcube2.win_func) == type(None):
            self.tfunc = self.mapcube1.win_func
        elif type(self.mapcube1.win_func) == type(None) and type(self.mapcube2.win_func) != type(None):
            self.tfunc = self.mapcube2.win_func
        else:
            self.tfunc = np.ones((3*self.nside))
        self.binned_mixing_matrix = np.einsum('ij,jk,kl', self.binning, self.unbinned_mixing_matrix, self.tfunc[:,None]*self.unbinning)

    def compute_pseudo_cl_cov_from_cl_cov(self, cl_cov, startl=2):
        if self.both_weights_separable:
            ell_size = np.shape(cl_cov)[0] 
            try:
                binned_mixing_matrix = self.binned_mixing_matrix
            except AttributeError:
                self.compute_binned_mixing_matrix(startl=startl)
            return np.einsum('ij, jklm, ln->iknm', self.binned_mixing_matrix[:ell_size,:ell_size], cl_cov, self.binned_mixing_matrix[:ell_size,:ell_size].T)

    def compute_ell_coupling(self):
        import pymaster as nmt
        map_dict1 = self.mapcube1.map_dict
        map_dict2 = self.mapcube2.map_dict
        if self.both_weights_separable:
            self.ell_coupling = nmt.NmtCovarianceWorkspace()
            #self.ell_coupling.compute_coupling_coefficients(self.cross_coupling_matrix,self.cross_coupling_matrix)
            self.ell_coupling.compute_coupling_coefficients(map_dict1['0'], map_dict2['0'], map_dict1['0'], map_dict2['0'])
        else:
            self.ell_coupling = {}
            for key in product(list(self.cross_coupling_matrix.keys()), list(self.cross_coupling_matrix.keys())):
                for key2 in product(list(self.cross_coupling_matrix.keys()), list(self.cross_coupling_matrix.keys())):
                    self.ell_coupling[key[0]+key[1]] = nmt.NmtCovarianceWorkspace()
                    #self.ell_coupling[key[0]+key[1]].compute_coupling_coefficients(self.cross_coupling_matrix[key[0]], self.cross_coupling_matrix[key[1]])
                    self.ell_coupling[key[0]+key[1]+key2[0]+key2[1]].compute_coupling_matrix(map_dict1[key[0]], map_dict2[key[1]], map_dict1[key2[0]], map_dict2[key2[1]])

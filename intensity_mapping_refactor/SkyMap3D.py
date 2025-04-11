import numpy as np
import scipy as sp
import healpy as hp
import h5py
import copy
from itertools import product
import pymaster as nmt

class Saveable_Loadable_hdf5(object):

    def write_to_hdf5(self, fname):
        with h5py.File(fname, 'w') as f:
            self.write_to_hdf5_group(f)

    def write_to_hdf5_group(self, group):
        for key in list(self.__dict__.keys()):
            if type(self.__dict__[key]) == type(None):
                group[key] = 'None'
            else:
                try:
                    write_func = self.__dict__[key].write_to_hdf5_group
                    group.create_group(key)
                    write_func(group[key])
                except AttributeError:
                    try:
                        group[key] = self.__dict__[key]
                    except TypeError as e:
                        print('Not saving attribute indexed by ' + key + ' because ' + str(e))
                        pass
    @classmethod
    def load_from_hdf5(cls, fname):
        with h5py.File(fname, 'r') as f:
            ans = cls.load_parameters_from_group(f)
        return ans

    @classmethod
    def load_parameters_from_group(cls, group):
        ans = cls()
        for key in group.keys():
            if not isinstance(group[key], h5py.Dataset):
                #print(key + ' is not a Dataset.')
                #print(group[key]['name'].asstr()[()])
                try:
                    sub_object_name = group[key]['name'].asstr()[()]
                    #print(sub_object_name)
                    setattr(ans, key, eval(sub_object_name).load_parameters_from_group(group[key]))
                except:
                    print('Failed to create attribute and underlying attribute structure for object indexed by ' + key)
            else:
                attr = np.array(group[key])
                if attr.size == 1:
                    attr = attr.item()
                if type(attr) != np.ndarray and attr == 'None':
                    setattr(ans, key, None)
                else:
                    setattr(ans, key, attr)
        return ans

class SphericalSkyMap3D(Saveable_Loadable_hdf5):

    def __init__(self, map_array = None, weights = None, z_edges = None, win_func = None):
        if type(map_array) != type(None) and type(weights) != type(None):
            self.standard_constructor(map_array, weights, z_edges = z_edges, win_func = win_func)
        else:
            pass

    def standard_constructor(self, map_array, weights, z_edges = None, win_func = None):
        if map_array.ndim != 1 and map_array.ndim != 2:
            raise TypeError("Map array must have dimension 1 or 2.")
        nside = (map_array.shape[-1]/12)**0.5
        if nside != int(nside):
            raise TypeError("Maps must be in HealPix format. Trailing dimension must have size 12*nside^2.")
        else:
            self.nside = int(nside)
        if map_array.ndim == 2:
            self.map_array = map_array
        else:
            self.map_array = map_array[np.newaxis,:]
        if weights.ndim != 1 and weights.ndim != 2:
            raise TypeError("Weights must have dimension 1 or 2.")
        if (weights.shape[-1]/12)**0.5 != self.nside:
            raise TypeError("Weights must match nside of maps.")
        if weights.ndim == 2:
            self.weights = weights
        else:
            self.weights = weights[np.newaxis,:]
        self.z_edges = z_edges
        if type(win_func) == type(None):
            self.win_func = None
        elif win_func.ndim != 1 and win_func.ndim != 2:
            raise TypeError("Window function must have dimension 1 or 2.")
        elif win_func.shape[-1] != 3*self.nside:
            raise TypeError("Window function must have trailing dimension 3*nside.")
        elif win_func.ndim == 2:
            self.win_func = win_func
        else:
            self.win_func = win_func[None,:]*np.ones((self.map_array.shape[0]))[:,None]
        self.templates = None

    @classmethod
    def from_gal_counts(cls, gal_count_array, sel_func, z_edges = None, win_func = None, make_sel_func_separable = True, cut_low_sel_func_voxels = True, cut_factor = 3., sel_func_normalized=True):
        #gal_count_array is the real galaxy number count.
        #sel_func is the expected unclustered galaxy count.
        if not sel_func_normalized:
                sel_func *= np.sum(gal_count_array)/np.sum(sel_func)
        if make_sel_func_separable:
            z_func = np.mean(sel_func, axis=1)
            theta_func = np.mean(sel_func, axis=0)
            sep_sel = z_func[:,None]*theta_func[None,:]
            sep_sel *= np.sum(sel_func)/np.sum(sep_sel)
            sel_func = sep_sel
            if cut_low_sel_func_voxels:
                #Cut values more than cut_factor standard deviations below mean.
                non_zero_bool = theta_func != 0
                mean = np.mean(theta_func[non_zero_bool])
                std = np.std(theta_func[non_zero_bool])
                cut_bool = (theta_func < mean - cut_factor*std)*np.ones(gal_count_array.shape, dtype=bool)
                gal_count_array[cut_bool] = 0.
                sel_func[cut_bool] = 0.
        non_zero_bool = sel_func != 0
        map_array = np.zeros(gal_count_array.shape)
        map_array[non_zero_bool] = gal_count_array[non_zero_bool]/sel_func[non_zero_bool] - 1.
        return cls(map_array, sel_func, z_edges, win_func)

    def change_nside(self, new_nside, update_win_func = True, weight_type = 'gal_selection'):
        if weight_type == 'gal_selection':
            #Keep sum of selection function the same.
            power_number = -2
        elif weight_type == 'inv_thermal_variance':
            #Keep noise variance times omega_pixel the same.
            power_number = 2
            self.weights = self.weights**-1.
        else:
            #Keep mean of noise the same.
            power_number = 0
        if type(self.win_func) != type(None):
            if update_win_func:
                self.win_func = self.win_func[:3*new_nside]*hp.sphtfunc.pixwin(new_nside)
        new_map_array = np.zeros((self.map_array.shape[0], 12*new_nside**2))
        for z in range(self.map_array.shape[0]):
            new_map_array[z,:] = hp.ud_grade(self.map_array[z,:], new_nside)
        new_weights = np.zeros((self.weights.shape[0], 12*new_nside**2))
        for z in range(self.weights.shape[0]):
            new_weights[z,:] = hp.ud_grade(self.weights[z,:], new_nside, power=power_number)
        self.nside = new_nside
        self.map_array = new_map_array
        self.weights = new_weights
        if weight_type == 'inv_thermal_variance':
            self.weights = self.weights**-1.

    def make_map_dict(self, subtract_weighted_mean = True, separable_weights = True, use_wmean_deprojection = False):
        self.map_dict = {}
        self.map_dict_no_win_func = {}
        self.separable_weights = separable_weights
        num_z = self.map_array.shape[0]
        for z in range(num_z):
            this_map = copy.deepcopy(self.map_array[z,:])
            if separable_weights:
                this_weight = self.weights[0,:]
            else:
                this_weight = self.weights[z,:]
            if type(self.win_func) == type(None):
                this_win_func = self.win_func
            else:
                this_win_func = self.win_func[z,:]
            if subtract_weighted_mean:
                if not use_wmean_deprojection:
                    this_map -= np.sum(this_weight*this_map)/np.sum(this_weight)
                    self.templates = None
                else:
                    self.templates = [[np.ones(this_map.shape)]]
            self.map_dict[str(z)]=nmt.NmtField(this_weight, [this_map], templates = self.templates, beam = this_win_func)
            self.map_dict_no_win_func[str(z)]=nmt.NmtField(this_weight, [this_map], templates = self.templates, beam = None)

class FlatSkyMap3D(Saveable_Loadable_hdf5):

    def __init__(self, map_array = None, weights = None, d_ra = None, d_dec = None, z_edges = None, win_func = None, ra_center = 0, dec_center = 0):
        if type(map_array) != type(None) and type(weights) != type(None):
            self.standard_constructor(map_array, weights, d_ra, d_dec, z_edges = z_edges, win_func = win_func, ra_center = ra_center, dec_center = dec_center)
        else:
            pass

    def standard_constructor(self, map_array, weights, d_ra, d_dec, z_edges = None, win_func = None, ra_center = 0, dec_center = 0):
        #Map_array and weights should have shape (n_z, n_ra, n_dec) or shape (n_ra, n_dec).
        #All angular values are expected to be right ascension and declination, in degrees.
        #Compute cos(dec) factor using dec_center.
        cos_dec = np.cos(np.deg2rad(dec_center))
        self.ra_center = ra_center
        self.dec_center = dec_center
        if map_array.ndim != 2 and map_array.ndim != 3:
            raise TypeError("Map array must have dimension 2 or 3.")
        if map_array.ndim == 3:
            self.map_array = map_array
        else:
            self.map_array = map_array[np.newaxis,:]
        #Store map size in number of pixels.
        self.n_ra = map_array.shape[-1]
        self.n_dec = map_array.shape[-2]
        #Compute and store size of sky patch in radians.
        self.length_ra = self.n_ra*cos_dec*np.deg2rad(d_ra)
        self.length_dec = self.n_dec*np.deg2rad(d_dec)
        if weights.ndim != 2 and weights.ndim != 3:
            raise TypeError("Weights must have dimension 2 or 3.")
        if weights.shape[-1] != self.n_ra or weights.shape[-2] != self.n_dec:
            raise TypeError("Angular dimensions of Weights must match that of maps.")
        if weights.ndim == 3:
            self.weights = weights
        else:
            self.weights = weights[np.newaxis,:]
        self.z_edges = z_edges
        if type(win_func) == type(None):
            self.win_func = None
        elif win_func.ndim != 1 and win_func.ndim != 2:
            raise TypeError("Window function must have dimension 1 or 2.")
        elif win_func.shape[-1] != 3*self.nside:
            raise TypeError("Window function must have trailing dimension 3*nside.")
        else:
            if win_func.ndim == 2:
                self.win_func = win_func
            else:
                self.win_func = win_func[None,:]*np.ones((self.map_array.shape[0]))[:,None]

    @classmethod
    def from_gal_counts(cls, gal_count_array, sel_func, d_ra, d_dec, z_edges = None, win_func = None, ra_center = 0, dec_center = 0, make_sel_func_separable = True, cut_low_sel_func_voxels = True, cut_factor = 3., sel_func_normalized=True):
        #gal_count_array is the real galaxy number count.
        #sel_func is the expected unclustered galaxy count.
        if not sel_func_normalized:
                sel_func *= np.sum(gal_count_array)/np.sum(sel_func)
        if make_sel_func_separable:
            z_func = np.mean(np.mean(sel_func, axis=1), axis=1)
            theta_func = np.mean(sel_func, axis=0)
            sep_sel = z_func[:,None,None]*theta_func[None,:,:]
            sep_sel *= np.sum(sel_func)/np.sum(sep_sel)
            sel_func = sep_sel
            if cut_low_sel_func_voxels:
                #Cut values more than cut_factor standard deviations below mean.
                non_zero_bool = theta_func != 0
                mean = np.mean(theta_func[non_zero_bool])
                std = np.std(theta_func[non_zero_bool])
                cut_bool = (theta_func < mean - cut_factor*std)*np.ones(gal_count_array.shape, dtype=bool)
                gal_count_array[cut_bool] = 0.
                sel_func[cut_bool] = 0.
        non_zero_bool = sel_func != 0
        map_array = np.zeros(gal_count_array.shape)
        map_array[non_zero_bool] = gal_count_array[non_zero_bool]/sel_func[non_zero_bool] - 1.
        return cls(map_array, sel_func, d_ra, d_dec, z_edges, win_func, ra_center, dec_center)

    def make_map_dict(self, subtract_weighted_mean = True, separable_weights = True, use_wmean_deprojection = False):
        self.map_dict = {}
        self.map_dict_no_win_func = {}
        self.separable_weights = separable_weights
        num_z = self.map_array.shape[0]
        for z in range(num_z):
            this_map = copy.deepcopy(self.map_array[z,:])
            if separable_weights:
                this_weight = self.weights[0,:]
            else:
                this_weight = self.weights[z,:]
            if type(self.win_func) == type(None):
                this_win_func = self.win_func
            else:
                this_win_func = self.win_func[z,:]
            if subtract_weighted_mean:
                if not use_wmean_deprojection:
                    this_map -= np.sum(this_weight*this_map)/np.sum(this_weight)
                    self.templates = None
                else:
                    self.templates = [[np.ones(this_map.shape)]]
            self.map_dict[str(z)]=nmt.NmtFieldFlat(self.length_ra, self.length_dec, this_weight, [this_map], templates = self.templates, beam = this_win_func)
            self.map_dict_no_win_func[str(z)]=nmt.NmtFieldFlat(self.length_ra, self.length_dec, this_weight, [this_map], templates = self.templates, beam = None)

class TomographicPair(Saveable_Loadable_hdf5):

    def __init__(self, maps1 = None, maps2 = None, force_different = False):
        if type(maps1) != type(None) and type(maps2) != type(None):
            self.standard_constructor(maps1 = maps1, maps2 = maps2, force_different = force_different)
        else:
            pass

    def standard_constructor(self, maps1, maps2, force_different=False):
        self.maps1 = maps1
        self.maps2 = maps2
        if self.maps1 is self.maps2:
            if not force_different:
                print("Maps1 and maps2 are the same object. Be sure this is an auto-power.")
            else:
                self.maps2 = copy.deepcopy(maps2)
        self.maps1.name = type(maps1).__name__
        self.maps2.name = type(maps2).__name__
        if maps1.map_array.ndim != maps2.map_array.ndim:
            raise TypeError("Tomographic map pairs do not have the same number of dimensions. Maybe one is flat sky and one is curved?")
        if maps1.map_array.ndim == 2:
            #Maps are curved sky.
            if maps1.map_array.shape[1] != maps2.map_array.shape[1]:
                raise TypeError("Curved sky map pairs do not have the same Nside.")
            self.nside = maps1.nside
            self.ell_max = 3*self.nside - 1
            self.flat_sky = False
        elif maps1.map_array.ndim == 3:
            #Maps are flat sky.
            if maps1.map_array.shape[1] != maps2.map_array.shape[1] or maps1.map_array.shape[2] != maps2.map_array.shape[2]:
                raise TypeError("Flat sky maps do not have same pixel dimensions.")
            if maps1.n_ra != maps2.n_ra or maps1.n_dec != maps2.n_dec or maps1.length_ra != maps2.length_ra or maps1.length_dec != maps2.length_dec:
                raise TypeError("Flat sky map pairs do not agree on either pixel number or angular scale.")
            else:
                self.n_ra = maps1.n_ra
                self.n_dec = maps1.n_dec
                self.n_max = max(self.n_ra, self.n_dec)
                self.length_ra = maps1.length_ra
                self.length_dec = maps1.length_dec
                self.ell_max = self.n_max*max(np.pi/self.length_ra, np.pi/self.length_dec)
                self.flat_sky = True
        else:
            raise TypeError("Tomographic map pairs do not have dimension 2 or 3.")

    def define_bins(self, bin_size = 16, is_Dell = False):
        if self.flat_sky:
            self.l0_bins = np.arange(self.n_max/bin_size)*bin_size*(self.ell_max/self.n_max)
            self.lf_bins = (np.arange(self.n_max/bin_size)+1)*bin_size*(self.ell_max/self.n_max)
            self.bins = nmt.NmtBinFlat(self.l0_bins, self.lf_bins)
        else:
            self.ell_bin_size = bin_size
            self.bins = nmt.NmtBin.from_nside_linear(self.nside, nlb=self.ell_bin_size)
        self.is_Dell = is_Dell

    def define_custom_bins(self, ell_lower_edges, ell_upper_edges, is_Dell = False):
        if self.flat_sky:
            self.l0_bins = ell_lower_edges
            self.lf_bins = ell_upper_edges
            self.bins = nmt.NmtBinFlat(self.l0_bins, self.lf_bins)
            self.is_Dell = False
        else:
            self.ell_bin_size = ell_upper_edges - ell_lower_edges
            self.bins = nmt.NmtBin.from_edges(ell_lower_edges, ell_upper_edges, is_Dell = is_Dell)
            self.is_Dell = is_Dell

    def Workspace(self):
        if self.flat_sky:
            return nmt.NmtWorkspaceFlat()
        else:
            return nmt.NmtWorkspace()

    def compute_coupled_cell(self, f1, f2):
        if self.flat_sky:
            return nmt.compute_coupled_cell_flat(f1, f2, self.bins)
        else:
            return nmt.compute_coupled_cell(f1, f2)

    def compute_correlation_matrices(self):
        f1 = self.maps1.map_dict['0']
        f2 = self.maps2.map_dict['0']
        if self.flat_sky:
            self.covariance_workspace = nmt.covariance.NmtCovarianceWorkspaceFlat()
            self.covariance_workspace.compute_coupling_coefficients(f1, f2, self.bins)
        else:
            self.covariance_workspace = nmt.covariance.NmtCovarianceWorkspace()
            self.covariance_workspace.compute_coupling_coefficients(f1, f2)
            

    def compute_mixing_matrix(self, bin_size = 16):
        try:
            bins = self.bins
        except AttributeError:
            self.define_bins(bin_size = bin_size)
            bins = self.bins
        self.mixing_matrix = {}
        if self.maps1.separable_weights and self.maps2.separable_weights:
            self.both_weights_separable = True
            #self.mixing_matrix = self.Workspace()
            #self.mixing_matrix.compute_coupling_matrix(self.maps1.map_dict['0'], self.maps2.map_dict['0'], bins)
            mixing_matrix = self.Workspace()
            mixing_matrix.compute_coupling_matrix(self.maps1.map_dict['0'], self.maps2.map_dict['0'], bins)
            for key in product(list(self.maps1.map_dict.keys()), list(self.maps2.map_dict.keys())):
                mixing_key = key[0] + '_' + key[1]
                self.mixing_matrix[mixing_key] = mixing_matrix
        else:
            #self.mixing_matrix = {}
            for key in product(list(self.maps1.map_dict.keys()), list(self.maps2.map_dict.keys())):
                mixing_matrix = self.Workspace()
                mixing_matrix.compute_coupling_matrix(self.maps1.map_dict[key[0]], self.maps2.map_dict[key[1]], bins)
                mixing_key = key[0] + '_' + key[1]
                self.mixing_matrix[mixing_key] = mixing_matrix

    def compute_cl_zz(self, weights1_separable = True, weights2_separable = True, bin_size = 16, save_unbinned_pcl = False):
        #print('1')
        #return
        try:
            map1_dict = self.maps1.map_dict
        except AttributeError:
            self.maps1.make_map_dict(separable_weights = weights1_separable)
        #print('1.5')
        try:
            map2_dict = self.maps2.map_dict
        except AttributeError:
            self.maps2.make_map_dict(separable_weights = weights2_separable)
        #print('2')
        try:
            mixing_matrix = self.mixing_matrix
        except AttributeError:
            try: 
                bin_size = self.ell_bin_size
            except AttributeError:
                pass
            #print(bin_size)
            self.compute_mixing_matrix(bin_size = bin_size)
            mixing_matrix = self.mixing_matrix
        #print('3')
        self.ells = np.array(self.bins.get_effective_ells())
        self.cl_zz = np.zeros((self.ells.size, self.maps1.map_array.shape[0], self.maps2.map_array.shape[0]))
        #print(self.cl_zz.shape)
        if save_unbinned_pcl:
            self.pcl_zz_unbinned = np.zeros((self.ell_max +1, self.maps1.map_array.shape[0], self.maps2.map_array.shape[0]))
            self.pcl_zz_binned = np.zeros(self.cl_zz.shape)
        for key in product(list(self.maps1.map_dict.keys()), list(self.maps2.map_dict.keys())):
            c_coupled=self.compute_coupled_cell(self.maps1.map_dict[key[0]],self.maps2.map_dict[key[1]])
            if save_unbinned_pcl:
                self.pcl_zz_unbinned[:,int(key[0]),int(key[1])] = c_coupled
                self.pcl_zz_binned[:,int(key[0]),int(key[1])] = self.bins.bin_cell(c_coupled)
            #if self.both_weights_separable:
            #    mixing_mat = self.mixing_matrix
            mixing_key = key[0] + '_' + key[1]
            #print(mixing_key)
            #print(mixing_matrix)
            current_mixing_mat = mixing_matrix[mixing_key]
            #print(current_mixing_mat)
            #print(int(key[0]))
            #print(int(key[1]))
            #print(current_mixing_mat.decouple_cell(c_coupled))
            #print(current_mixing_mat.decouple_cell)
            #print(c_coupled)
            #print(c_coupled.shape)
            #print(np.squeeze(c_coupled).shape)
            self.cl_zz[:,int(key[0]),int(key[1])]=current_mixing_mat.decouple_cell(c_coupled)

    def bin_cl_model_ell_only(self, cl_model, forward_model = False, simulate_weighted_mean_subt=False, pcl_correction = None, bin_forward_model = True):
        #Assuming weights are fully separable.
        key = list(self.mixing_matrix.keys())[0]
        ans = self.mixing_matrix[key].couple_cell([cl_model])
        if not forward_model:
            ans = self.mixing_matrix[key].decouple_cell(ans)
        ans = ans[0,:]
        if forward_model and bin_forward_model:
            ans = self.bins.bin_cell(ans)
        return ans

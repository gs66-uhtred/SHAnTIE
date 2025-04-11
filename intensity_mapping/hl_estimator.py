#This module will implement a Hamimeche & Lewis estimator class.

from copyreg import pickle
from types import MethodType

import numpy as np
from . import pwrspec_estimation as pe
import numba
from numba import jit, njit
import scipy
import copy 

#def ln_posterior(params, likelihood_object, model_func, ln_prior_function = None):
#    if type(ln_prior_function) != type(None):
#        ans = likelihood_object.ln_like(model_func, params)+ln_prior_function(arr)
#    else:
#        ans = likelihood_object.ln_like(model_func, params)
#    return ans

def ln_posterior(arr, likelihood_object):
    if type(likelihood_object.ln_prior_function) == type(None):
        return likelihood_object.ln_like(likelihood_object.model_func, arr)
    elif likelihood_object.ln_prior_function(arr) == -np.inf:
        return -np.inf
    else:
        return likelihood_object.ln_like(likelihood_object.model_func, arr)+likelihood_object.ln_prior_function(arr)

def execute_mcmc(EnsembleSampler, params_guess, ndim, nwalkers = 100, run_length = 100, randomize_scale = 1e-2):
        params_guess = np.array(params_guess)
        pos = [params_guess + params_guess*randomize_scale*np.random.randn(ndim) for i in range(nwalkers)]
        #print(pos)
        pos, prob, state = EnsembleSampler.run_mcmc(pos, run_length)
        return pos, prob, state

def execute_2nd_mcmc(EnsembleSampler, old_mcmc_chain, ndim, nwalkers = 100, run_length = 100):
        rand_int = np.random.randint(0, high=old_mcmc_chain.shape[0], size=nwalkers, dtype=int)
        params_guess = old_mcmc_chain[rand_int,:]
        pos, prob, state = EnsembleSampler.run_mcmc(params_guess, run_length)
        return pos, prob, state

def initialize_mcmc(ln_posterior, ndim , nwalkers = 100, pool = None, args = [], threads = 4, ln_prior_function = None):
        import emcee
        EnsembleSampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, pool = pool, args=args, threads=threads)
        return EnsembleSampler

#Define useful functions for HL estimation.
#Make hl_likelihood class to compute likelihood. Should precompute fiducial quantities and hold as attributes.

def g(evals):
    return np.sign(evals - 1)*(2*(evals - np.log(evals) - 1))**0.5

def inv_sqrt(mat, no_inverse=False):
    #mat should be symmetric.
    evals_vecs = np.linalg.eigh(mat)
    new_evals = np.zeros((evals_vecs[0].shape[0],evals_vecs[0].shape[1],evals_vecs[0].shape[1]))
    for ind in range(evals_vecs[0].shape[0]):
        if no_inverse:
            new_evals[ind,:] = np.diag(evals_vecs[0][ind,:]**0.5)
        else:
            new_evals[ind,:] = np.diag(evals_vecs[0][ind,:]**-0.5)
    return np.einsum('ijk,ikl,iml->ijm',evals_vecs[1], new_evals, evals_vecs[1])

def g_mat(mat):
    #mat should be symmetric.
    eig_decomp = np.linalg.eig(mat)
    vect_inv = np.linalg.inv(eig_decomp[1])
    eval_mat = np.zeros((eig_decomp[0].shape[0],eig_decomp[0].shape[1],eig_decomp[0].shape[1]))
    evals_g = g(eig_decomp[0])
    for ind in range(eig_decomp[0].shape[0]):
        eval_mat[ind,:,:] = np.diag(evals_g[ind,:])
    ans = np.einsum('ijk,ikl,ilm->ijm', eig_decomp[1], eval_mat, vect_inv)
    return ans

class likelihood_estimator(object):

    def ln_like(self):
        #To be overwritten by specific estimator.
        return -np.inf

    #def ln_posterior(self, arr):
    #    if type(self.ln_prior_function) == type(None):
    #        return self.ln_like(self.model_func, arr)
    #    else:
    #        return self.ln_like(self.model_func, arr)*self.ln_prior_function(arr)

    def initialize_mcmc(self, model_func, ndim , nwalkers = 100, args = [], threads = 4, ln_prior_function = None):
        import emcee
        self.nwalkers = nwalkers
        self.threads = threads
        self.ndim = ndim
        self.model_func = model_func
        self.args = args
        self.ln_prior_function = ln_prior_function
        #if type(ln_prior_function) == type(None):
        #    self.ln_posterior = lambda arr:self.ln_like(self.model_func, arr)
        #else:
        #    self.ln_posterior = lambda arr:self.ln_like(self.model_func, arr)+ln_prior_function(arr)
        #def wrapper(arr):
        #    return self.ln_posterior(arr)
        #EnsembleSampler = emcee.EnsembleSampler(nwalkers, ndim, wrapper, args=args, threads=threads)
        EnsembleSampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=self, threads=threads)
        return EnsembleSampler

    def set_model_and_prior(self, model_func, ln_prior_function = None):
        self.model_func = model_func
        self.ln_prior_function = ln_prior_function

    def execute_mcmc(self, EnsembleSampler, params_guess, run_length, randomize_scale = 1e-2):
        params_guess = np.array(params_guess)
        self.pos = [params_guess + params_guess*randomize_scale*np.random.randn(self.ndim) for i in range(self.nwalkers)]
        self.pos, self.prob, self.state = EnsembleSampler.run_mcmc(self.pos, run_length)
        #pos, prob, state = EnsembleSampler.run_mcmc(self.pos, run_length) 

    def run_ml_estimation(self, params_guess, model_func = None, ln_prior_function = None, method = 'BFGS', constraints=()):
        if type(model_func) != type(None):
            self.model_func = model_func
        if type(ln_prior_function) != type(None):
            self.ln_prior_function = ln_prior_function
        import scipy.optimize as op
        nll = lambda arr:-ln_posterior(arr, self)
        result = op.minimize(nll, params_guess, method = method, constraints = constraints)
        return result

class hl_likelihood(likelihood_estimator):

    #cl should be in (ell,z,z') shape.
    #fid_cov should be in (ell,z,z',ell',z'',z''') shape.
    #cl_range gives [ell_min_index,ell_max_index+1,z_min_index,z_max_index+1] that will be considered.
    cl_range = None
    fid_cl = None
    fid_cov = None
    cl_dat = None
    offset = 0
    identity = None
    fid_cl_sqrt = None
    #If num_cross_pairs > 1, then z index repeats that many times. Set to 2 for full FIRAS and BOSS analysis. 
    num_cross_pairs = 1

    def __init__(self, cl_dat=None, fid_cl=None, fid_cov=None, cl_range = None, offset=0, num_cross_pairs = 1):
        self.cl_dat = cl_dat
        self.fid_cl = fid_cl
        self.fid_cov = fid_cov
        self.offset = offset
        if type(cl_range) == type(None) and type(cl_dat) != type(None):
            self.cl_range = [0,cl_dat.shape[0],0,cl_dat.shape[1]]
        elif type(cl_range) != type(None) and num_cross_pairs>1:
            #If more than 1 map type, do not try to trim z_index, which repeats.
            self.cl_range = [0,cl_range[1],0, cl_dat.shape[1]]
        else:
            self.cl_range = cl_range

    def precompute(self, data_only=False):
        ell_min = self.cl_range[0]
        ell_max = self.cl_range[1] - 1
        ell_size = ell_max - ell_min + 1
        z_min = self.cl_range[2]
        z_max = self.cl_range[3] - 1
        z_size = z_max - z_min + 1
        self.identity = np.ones((ell_size,z_size,z_size))*np.eye(z_size)
        if type(self.cl_dat) != type(None):
            print("Creating slice into specified part of cl data.")
            self.cl_dat_limited = self.cl_dat[ell_min:ell_max+1,z_min:z_max+1,z_min:z_max+1]
        if type(self.fid_cl) != type(None) and data_only == False:
            print("Computing square root of fiducial model.")
            fid_cl_limited = self.fid_cl[ell_min:ell_max+1,z_min:z_max+1,z_min:z_max+1]
            self.fid_cl_sqrt = inv_sqrt(fid_cl_limited + self.offset*self.identity, no_inverse = True)
        if type(self.fid_cov) != type(None) and data_only == False:
            print("Computing inverse of vectorized, flattened fiducial model")
            z_pair_size = z_size*(z_size+1)/2
            cov_vec = np.zeros((ell_size, z_pair_size, ell_size, z_pair_size))
            cov_limited = self.fid_cov[ell_min:ell_max+1,z_min:z_max+1,z_min:z_max+1,ell_min:ell_max+1,z_min:z_max+1,z_min:z_max+1]
            self.cov_limited = cov_limited
            for ell in range(ell_size):
                fid_vec, cov_vec[:,:,ell,:] = pe.truncate_clmat_cov(fid_cl_limited, cov_limited[:,:,:,ell,:,:])
            cov_vec_flat = cov_vec.reshape((ell_size*z_pair_size,ell_size*z_pair_size))
            self.fid_cov_inv = np.linalg.inv(cov_vec_flat)

    def intermediate_data_amp(self, model_func, model_params):
        cl_model = model_func(model_params, self.cl_range)
        cl_model += self.offset*self.identity
        inv_sqrt_ = inv_sqrt(cl_model)
        ans = np.einsum('ijk,ikl,ilm->ijm',inv_sqrt_, self.cl_dat_limited + self.offset*self.identity, inv_sqrt_)
        return ans

    def amp_data_mat(self, model_func, model_params):
        middle = g_mat(self.intermediate_data_amp(model_func, model_params))
        ans = np.einsum('ijk,ikl,ilm->ijm', self.fid_cl_sqrt, middle, self.fid_cl_sqrt)
        return ans

    def ln_like(self, model_func, model_params):
        dat_mat = self.amp_data_mat(model_func, model_params)
        dat_vect, useless = pe.truncate_clmat_cov(dat_mat, self.cov_limited[:,:,:,0,:,:])
        dat_vect_flat = dat_vect.reshape((dat_vect.size))
        ans = -0.5*np.einsum('i,ij,j', dat_vect_flat, self.fid_cov_inv, dat_vect_flat)
        return ans

class hl_likelihood2(likelihood_estimator):

    #cl should be in (ell,z,z') shape.
    #fid_cov should be in (ell,z_combinations,ell',z_combinations) shape.
    #Full cl range will be considered for fiducial model, covariance, and data (cuts must be beforehand).
    #cl_range will be applied to model fit.
    fid_cl = None
    fid_cov = None
    cl_dat = None
    offset = 0
    identity = None
    fid_cl_sqrt = None

    def __init__(self, cl_dat=None, fid_cl=None, fid_cov=None, cl_range=None, offset=0):
        self.cl_dat = cl_dat
        self.fid_cl = fid_cl
        self.fid_cov = fid_cov
        self.offset = offset
        self.cl_range = cl_range
        if type(cl_range) == type(None):
            self.cl_range = [0,fid_cl.shape[0], 0, fid_cl.shape[1]]
        #self.cl_dat = cl_dat[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        #self.fid_cl = fid_cl[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]

    def precompute(self):
        ell_size = self.fid_cov.shape[0]
        z_size = self.cl_dat.shape[-1]
        z_tot_size = int((z_size+1)*z_size/2)
        if z_tot_size != self.fid_cov.shape[-1] or ell_size != self.cl_dat.shape[0]:
            raise ValueError('Cl data and fiducial covariance have incompatible shapes.')
        self.identity = np.ones(self.cl_dat.shape)*np.eye(z_size)
        if type(self.fid_cl) != type(None):
            print("Computing square root of fiducial model.")
            self.fid_cl_sqrt = inv_sqrt(self.fid_cl + self.offset*self.identity, no_inverse = True)
        if type(self.fid_cov) != type(None):
            print("Computing inverse of vectorized, flattened fiducial model")
            self.fid_cov_inv = np.linalg.pinv(self.fid_cov.reshape(ell_size*z_tot_size, ell_size*z_tot_size))
        print("Computing data z-indeces for redundant truncation of data. Assuming covariance ordering from cov_py method.")
        self.ind1_arr = np.zeros((z_tot_size), dtype=np.int)
        self.ind2_arr = np.zeros((z_tot_size), dtype=np.int)
        counter = 0
        for ind2 in range(z_size):
            for ind1 in range(ind2+1):
                self.ind1_arr[counter] = ind1
                self.ind2_arr[counter] = ind2
                counter += 1

    def intermediate_data_amp(self, model_func, model_params):
        cl_model = model_func(model_params, self.cl_range)
        cl_model += self.offset*self.identity
        inv_sqrt_ = inv_sqrt(cl_model)
        ans = np.einsum('ijk,ikl,ilm->ijm',inv_sqrt_, self.cl_dat + self.offset*self.identity, inv_sqrt_)
        return ans

    def amp_data_mat(self, model_func, model_params):
        middle = g_mat(self.intermediate_data_amp(model_func, model_params))
        ans = np.einsum('ijk,ikl,ilm->ijm', self.fid_cl_sqrt, middle, self.fid_cl_sqrt)
        return ans

    def ln_like(self, model_func, model_params):
        dat_mat = self.amp_data_mat(model_func, model_params)
        dat_vect = dat_mat[:,self.ind1_arr,self.ind2_arr].flatten()
        ans = -0.5*np.einsum('i,ij,j', dat_vect, self.fid_cov_inv, dat_vect)
        return ans

class gauss_approx_partialsky(likelihood_estimator):
    #Equation 29 of Hamimeche and Lewis. Use fiducial model for covariance.

    #range to consider in cl data: [z_start, z_end, l_start, l_end].
    cl_range = None
    #Holds measured cl or pseudo-cl.
    cl_data = None
    #Fiducial covariance.
    fid_cov = None
    #Boolean that is true if using fiducial model for covariance.
    fid_bool = True
    #Boolean on whether to use inverse covariance or linear solve.
    use_inv_cov = False

    def __init__(self, cl_data=None, cl_range=None, fid_cov = None, no_ln_det = True, fid_cov_sparse=False):
        self.cl_data = cl_data
        self.cl_range = cl_range
        self.cl_data_limited = self.cl_data[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        self.fid_cov_sparse = fid_cov_sparse
        #if type(fid_cov) != type(None) and not fid_cov_sparse:
        #    self.fid_cov = fid_cov[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3],cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        #    self.cov_reshape = np.einsum('ijklmn->iljkmn', self.fid_cov)
        #    self.cl_data_flat, temp_cov = pe.truncate_clmat_cov(self.cl_data_limited, self.cov_reshape)
        #    self.cl_data_flat = self.cl_data_flat.flatten()
        #    flatten_length = self.cl_data_flat.size
        #    self.cov_matrix_flat = np.einsum('ijkl->ikjl',temp_cov).reshape(flatten_length,flatten_length)
        #    #Now try to invert the covariance matrix.
        #    self.fid_cov_inv = np.linalg.pinv(self.cov_matrix_flat)
        #    self.use_inv_cov = True
        #    if not no_ln_det:
        #        self.ln_det = np.log(np.linalg.det(self.cov_matrix_flat))
        #        if np.isinf(self.ln_det):
        #            print 'Infinite log determinant of covariance. Setting log determinant to zero instead.'
        #            self.ln_det = 0.
        #    else:
        #        self.ln_det = 0.
        #elif type(fid_cov) == type(None):
        #    self.fid_bool = False
        if type(fid_cov) == type(None):
            self.fid_bool = False
        else:
            #If fid_cov is sparse, assume it is already trimmed appropriately in z (and flattened with no redundant elements)
            self.fid_cov = fid_cov
            self.cov_matrix_flat = fid_cov
            #Find flattened array of non-redundant Cl data.
            z_size = cl_range[3] - cl_range[2]
            self.ind1_arr = np.zeros((int((z_size+1)*z_size/2)), dtype=np.int)
            self.ind2_arr = np.zeros((int((z_size+1)*z_size/2)), dtype=np.int)
            counter = 0
            if fid_cov.ndim == 4:
                #Covariance is not sparse. Need to make flat version.
                #Does not currently support cutting z-range at this point. Must be done prior to covariance construction if desired.
                new_cov_height = fid_cov.shape[0]*fid_cov.shape[1]
                self.cov_matrix_flat = fid_cov.reshape((new_cov_height, new_cov_height))
                #Now try to invert the covariance matrix.
                self.fid_cov_inv = np.linalg.pinv(self.cov_matrix_flat)
                self.use_inv_cov = True
            for ind2 in range(z_size):
                for ind1 in range(ind2+1):
                    self.ind1_arr[counter] = ind1
                    self.ind2_arr[counter] = ind2
                    counter += 1
            self.cl_data_flat = self.cl_data_limited[:,self.ind1_arr,self.ind2_arr].flatten()
            if not no_ln_det:
                #print self.cov_matrix_flat.shape
                self.ln_det = np.sum(np.log(np.linalg.eig(self.cov_matrix_flat)[0]))
                if np.isinf(self.ln_det):
                    print('Infinite log determinant of covariance. Setting log determinant to zero instead.')
                    self.ln_det = 0.
            else:
                self.ln_det = 0.
        self.num_factor = self.cl_data_flat.size*np.log(2.*np.pi)

    def ln_like(self, model_func, model_params):
        self.current_model_params = model_params
        #if not self.fid_cov_sparse:
        #    flat_model = pe.truncate_clmat_cov(model_func(model_params, self.cl_range),self.cov_reshape)[0].flatten()
        #else:
        #    flat_model = model_func(model_params, self.cl_range)[self.cl_range[0]:self.cl_range[1],self.ind1_arr,self.ind2_arr].flatten()
        flat_model = model_func(model_params, self.cl_range)[:,self.ind1_arr,self.ind2_arr].flatten()
        #print flat_model.shape
        diff = self.cl_data_flat - flat_model
        self.diff = diff
        self.flat_model = flat_model
        if self.use_inv_cov:
            fact1 = np.einsum('i,ij,j', diff, self.fid_cov_inv, diff)
        else:
            #print diff.shape
            #print self.cov_matrix_flat.shape
            #print self.ln_det
            fact1 = np.dot(diff, scipy.sparse.linalg.cg(self.cov_matrix_flat, diff, tol=5*10**-2)[0])
        #print fact1
        self.chi_squared = fact1
        return -0.5*(fact1+self.ln_det + self.num_factor)

    def ln_like_partial_derivative(self, model_func, model_params):
        flat_model = model_func(model_params, self.cl_range)[:,self.ind1_arr,self.ind2_arr].flatten()
        diff = self.cl_data_flat - flat_model
        #diff_partial_derivative = -1.*np.ones(flat_model.shape)
        fact1 = -np.dot(self.fid_cov_inv, diff)
        #else:
        #    fact1 = np.dot(diff, scipy.sparse.linalg.cg(self.cov_matrix_flat, diff, tol=5*10**-2)[0])
        return fact1

    def jacobian_vector(self, model_func, model_params, model_derivative):
        #model_derivative should be a function whose answer has the shape (cl_model with cl_range applied, params_dimension).
        partial_derivative = self.ln_like_partial_derivative(model_func, model_params)
        current_model_derivative = model_derivative(model_params, self.cl_range)[:,self.ind1_arr,self.ind2_arr,:]
        current_model_derivative_reshaped = current_model_derivative.reshape(partial_derivative.size, len(model_params))
        return np.dot(partial_derivative, current_model_derivative_reshaped)
        

 
class gauss_approx_fullsky(likelihood_estimator):
    #Equation 30 of Hamimeche and Lewis. User can choose whether to use a fiducial model, as they do in the paper.

    #range to consider in cl data: [z_start, z_end, l_start, l_end].
    cl_range = None
    #Holds measured cl or pseudo-cl.
    cl_data = None
    #Number of moders per ell index. This is 2l+1 for spherical sky, but not for flat sky.
    num_modes = None    

    def __init__(self, cl_data=None, cl_range=None, num_modes=None, fid_model = None):
        self.cl_data = cl_data
        self.cl_range = cl_range
        self.num_modes = num_modes
        if type(cl_range) != type(None):
            self.cl_data_limited = cl_data[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            self.num_modes = num_modes[cl_range[0]:cl_range[1]]
        else:
            self.cl_data_limited = cl_data
        if type(fid_model) == type(None):
            self.use_fid = False
        else:
            self.fid_model = fid_model[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            self.fid_inv = np.linalg.inv(self.fid_model)
            self.use_fid = True

    
    def ln_like_fast(self, model_func, model_params):
        @njit
        def critical_math1(n, cl_model, cl_data, fid_model, fid_inv, num_modes):
            fact2 = (n+1)*np.sum(np.log(np.linalg.det(fid_model)))
            intermediate1 = np.einsum('ijk,ikl,ilm,imn->ijn', cl_model - cl_data, fid_inv, cl_model - cl_data, fid_inv)
            fact1 = np.sum(num_modes*np.trace(intermediate1, axis1=1, axis2=2)/2.)
            return -0.5*(fact1 + fact2)

        @njit
        def critical_math2(n, cl_model, cl_data, num_modes):
            fid_inv = np.zeros(cl_model.shape)
            for ell in range(cl_model.shape[0]):
                fid_inv[ell,:] = np.linalg.inv(cl_model[ell,:])
            #fid_inv = np.linalg.inv(cl_model)
            det_cl_model = np.zeros(cl_model.shape[0])
            for ell in range(cl_model.shape[0]):
                det_cl_model[ell] = np.linalg.det(cl_model[ell,:])
            #fact2 = (n+1)*np.sum(np.log(np.linalg.det(cl_model)))
            fact2 = (n+1)*np.sum(np.log(det_cl_model))
            intermediate1 = np.einsum('ijk,ikl,ilm,imn->ijn', cl_model - cl_data, fid_inv, cl_model - cl_data, fid_inv)
            fact1 = np.sum(num_modes*np.trace(intermediate1, axis1=1, axis2=2)/2.)
            return -0.5*(fact1 + fact2)


        cl_model = model_func(model_params, self.cl_range)
        n = self.cl_range[3] - self.cl_range[2]
        if self.use_fid:
            return critical_math1(n, cl_model, self.cl_data_limited, self.fid_model, self.fid_inv, self.num_modes)
        else:
            return critical_math2(n, cl_model, self.cl_data_limited, self.num_modes)

    def ln_like(self, model_func, model_params, lin_alg_solve = True):
        cl_model = model_func(model_params, self.cl_range)
        n = self.cl_range[3] - self.cl_range[2]
        self.n_ell = self.cl_range[1] - self.cl_range[0]
        fact3 = self.n_ell*n*np.log(2.) - np.sum(np.log(self.num_modes))*n*(n+1)/2. + self.n_ell*n*((n+1)/2)*np.log(2.*np.pi)
        if self.use_fid:
            fact2 = (n+1)*np.sum(np.log(np.linalg.det(self.fid_model)))
            #fact2 = (n+1)*np.sum(np.log(np.linalg.eigh(self.fid_model)[0]))
            intermediate1 = np.einsum('ijk,ikl,ilm,imn->ijn', cl_model - self.cl_data_limited, self.fid_inv, cl_model - self.cl_data_limited, self.fid_inv)
            fact1 = np.sum(self.num_modes*np.trace(intermediate1,axis1=1,axis2=2)/2.)
            #print fact1
            #print fact2 + fact3
            return -0.5*(fact1 + fact2 + fact3)
        elif not lin_alg_solve:
            cl_model_inv = np.linalg.inv(cl_model)
            fact2 = (n+1)*np.sum(np.log(np.linalg.det(cl_model)))
            #fact2 = (n+1)*np.sum(np.log(np.linalg.eigh(cl_model)[0]))
            intermediate1 = np.einsum('ijk,ikl,ilm,imn->ijn', cl_model - self.cl_data_limited, cl_model_inv, cl_model - self.cl_data_limited, cl_model_inv)
            #print intermediate1.shape
            #print np.trace(intermediate1, axis1=1, axis2=2).shape
            #print self.num_modes.shape
            fact1 = np.sum(self.num_modes*np.trace(intermediate1,axis1=1,axis2=2)/2.)
            return -0.5*(fact1 + fact2 + fact3)
        else:
            fact2 = (n+1)*np.sum(np.log(np.linalg.det(cl_model)))
            #fact2 = (n+1)*np.sum(np.log(np.linalg.eigh(cl_model)[0]))
            intermediate1 = np.linalg.solve(cl_model, cl_model - self.cl_data_limited)
            fact1 = np.sum(self.num_modes*np.trace(np.einsum('ijk,ikl->ijl',intermediate1,intermediate1),axis1=1,axis2=2)/2.)
            #print -0.5*fact2
            #print -0.5*fact1
            return -0.5*(fact1 + fact2 + fact3)

class gaussian_approx_cross(likelihood_estimator):
    ell_coupling=True
    cov_matrix=None
    auto=False
    cov_cl_model=None

    def __init__(self, cl_data = None, cl_range = None, fid_model_cov = None, cii_cov_piece = None, cib_cov_piece = None, cib_x_cii_cov_piece = None, use_det = False):
        self.cl_data = cl_data
        self.cl_range = cl_range
        if type(cl_range) != type(None):
            self.cl_data_limited = cl_data[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
        else:
            self.cl_data_limited = cl_data
        if type(fid_model_cov) == type(None):
            self.use_fid = False
        elif fid_model_cov.ndim==5:
            self.ell_coupling=False
            print(fid_model_cov.shape)
            self.fid_mod_cov = fid_model_cov[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            self.fid_mod_cov_flat = self.fid_mod_cov.reshape((cl_range[2]-cl_range[1],(cl_range[3]-cl_range[2])**2,(cl_range[3]-cl_range[2])**2))
            #For fiducial model, can set ln determinant to 0 since it is just an additive constant.
            self.ln_det=0
            self.fid_inv = np.linalg.pinv(self.fid_mod_cov_flat)
            del self.fid_mod_cov_flat
            del self.fid_mod_cov
            self.use_fid = True
        else:
            self.fid_mod_cov = fid_model_cov[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3],cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            self.fid_mod_cov_flat = self.fid_mod_cov.reshape((self.cl_data_limited.size, self.cl_data_limited.size))
            del self.fid_mod_cov
            #self.ln_det = np.log(np.linalg.det(self.fid_mod_cov_flat))
            #if np.isinf(self.ln_det):
            #For fiducial model, can set the ln determinant to 0 since it is just an additive constant.
            self.ln_det=0
            self.fid_inv = np.linalg.pinv(self.fid_mod_cov_flat)
            #del self.fid_mod_cov_flat
            self.use_fid = True
        if type(cii_cov_piece) != type(None) and type(cib_cov_piece) != type(None) and type(cib_x_cii_cov_piece) != type(None):
            self.update_cov_model = True
            self.cii_cov_piece = cii_cov_piece[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3],cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            self.cii_cov_piece_flat = self.cii_cov_piece.reshape((self.cl_data_limited.size,self.cl_data_limited.size))
            self.cib_cov_piece = cib_cov_piece[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3],cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            self.cib_cov_piece_flat = self.cib_cov_piece.reshape((self.cl_data_limited.size,self.cl_data_limited.size))
            self.cib_x_cii_cov_piece = cib_x_cii_cov_piece[cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3],cl_range[0]:cl_range[1],cl_range[2]:cl_range[3],cl_range[2]:cl_range[3]]
            self.cib_x_cii_cov_piece_flat = self.cib_x_cii_cov_piece.reshape((self.cl_data_limited.size,self.cl_data_limited.size))
            self.use_det = use_det
            #if use_det:
            #    self.ln_det = np.log(np.linalg.det(self.fid_mod_cov_flat))
            #    self.ln_det_cii_piece = np.log(np.linalg.det(self.cii_cov_piece_flat))
            #    self.ln_det_cib_piece = np.log(np.linalg.det(self.cib_cov_piece_flat))
        else:
            self.update_cov_model = False
            del self.fid_mod_cov_flat

    def pass_cov_from_model(self, cov_cl_model, precompute=False, precompute_args = None):
        #For case without fiducial model.
        #The covariance will depend on the parameters and will be calculated by a cl model object.
        self.cov_cl_model = cov_cl_model
        self.compute_cov = True
        if precompute:
           self.precompute_cov_pieces = True
           self.cov_cl_model.compute_cov_pieces(precompute_args[0], precompute_args[1], self.cl_range)

    def flatten(self):
        #Flatten data and covariance.
        if self.ell_coupling==True:
             if self.auto==False:
                 self.cl_data_flat = self.cl_data_limited.flatten()
                 flatten_length = self.cl_data_flat.size
             if not self.use_fid and type(self.cov_matrix) != type(None):
                 #self.cov_matrix_flat = self.cov_matrix.reshape(self.cov_matrix.shape[0], self.cov_matrix.shape[1], flatten_length, flatten_length)
                 if self.auto == False:
                     self.cov_matrix_flat = self.cov_matrix.reshape(flatten_length,flatten_length)
                 else:
                     #If this is actually an auto-power, need to remove redundant frequencies.
                     self.cov_reshape = np.einsum('ijklmn->iljkmn', self.cov_matrix)
                     self.cl_data_flat, temp_cov = pe.truncate_clmat_cov(self.cl_data_limited, self.cov_reshape)
                     self.cl_data_flat = self.cl_data_flat.flatten()
                     flatten_length = self.cl_data_flat.size
                     self.cov_matrix_flat = np.einsum('ijkl->ikjl',temp_cov).reshape(flatten_length,flatten_length)
        else:
            self.cl_data_flat = self.cl_data_limited.reshape((self.cl_range[2]-self.cl_range[1],(self.cl_range[3]-self.cl_range[2])**2))

    def ln_like(self, model_func, model_params):
        if not self.auto:
            self.flatten()
            if self.ell_coupling:
                flat_model = model_func(model_params, self.cl_range).flatten()
            else:
                flat_model = model_func(model_params, self.cl_range).reshape((self.cl_range[2]-self.cl_range[1],(self.cl_range[3]-self.cl_range[2])**2))
            diff = flat_model - self.cl_data_flat
        if not self.use_fid and self.auto==True:
            self.cov_matrix = self.cov_cl_model.finalize_cov(model_params, self.cl_range)
            self.flatten()
            flat_model = pe.truncate_clmat_cov(model_func(model_params, self.cl_range),self.cov_reshape)[0].flatten()
            diff = flat_model - self.cl_data_flat
            #cov = np.einsum('i,j,ijkl', model_params, model_params, self.cov_matrix_flat)
            cov = self.cov_matrix_flat
            fact2 = np.log(np.linalg.det(cov))
            if np.isinf(fact2):
                fact2=0
            fact1 = np.einsum('i,i', diff, np.linalg.solve(cov, diff))
            return -0.5*(fact1+fact2)
        elif self.ell_coupling:
            if not self.update_cov_model:
                fact1 = np.einsum('i,ij,j', diff, self.fid_inv, diff)
                return -0.5*(fact1+self.ln_det)
            else:
                #Construct covariance with correct parameters.
                #Assume model_params[0] is bias*CIB amplitude and model_params[2] is bias*CII amplitude.
                cov = self.fid_mod_cov_flat + model_params[0]**2*self.cib_cov_piece_flat + model_params[2]**2*self.cii_cov_piece_flat + model_params[0]*model_params[2]*self.cib_x_cii_cov_piece_flat
                intermediate1 = np.linalg.solve(cov, diff)
                fact1 = np.dot(diff, intermediate1)
                if self.use_det:
                    #self.ln_det = np.log(np.linalg.det(cov))
                    self.ln_det = np.linalg.slogdet(cov)[1]
                    return -0.5*(fact1 + self.ln_det)
                else:
                    return -0.5*fact1 
        else:
            fact1 = np.einsum('ki,kij,kj', diff, self.fid_inv, diff)
            return -0.5*(fact1+self.ln_det)


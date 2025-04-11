#Module for code that generates Cl(z,z') covariance models.
#Most functions will require pymaster to compute ell,ell' coupling. 

import numpy as np
import scipy
import pymaster as nmt

def cov_repeats(ell_coupling, workspace, cl_models, ell_size, sparse_ell = False, ell_diags = [0,1,-1]):
    #Ell_coupling should be NaMaster NmtCovarianceWorkspace object with coupling coefficients calculated.
    #Cl models should be a list of 4 Cl models of dimension [ell, z, z].
    #Set sparse_ell to True to use scipy sparse matrix and ignore ell correlations between distant ell-bins.
    #ell_diags sets the diagonals and off-diagonals to include in the sparse approximation of the ell coupling matrix.
    z_size = cl_models[0].shape[-1]
    if not sparse_ell:
        cov_output = np.zeros((ell_size, z_size, z_size, ell_size, z_size, z_size))
    if sparse_ell:
        sparse_values = []
        sparse_col = []
        sparse_row = []
    for ind1 in range(z_size):
        for ind2 in range(z_size):
            for ind3 in range(z_size):
                for ind4 in range(z_size):
                    cl_auto1 = cl_models[0]
                    cl_cross12 = cl_models[1]
                    cl_cross21 = cl_models[2]
                    cl_auto2 = cl_models[3]
                    cov_temp = nmt.gaussian_covariance(ell_coupling, 0, 0, 0, 0, [cl_auto1[:,ind1,ind3]],[cl_cross12[:,ind1,ind4]],[cl_cross21[:,ind2,ind3]],[cl_auto2[:,ind2,ind4]], workspace, wb=workspace)[:ell_size, :ell_size]
                    #cov_temp = nmt.gaussian_covariance(ell_coupling,cl_auto1[:,ind1,ind3],cl_cross12[:,ind1,ind4],cl_cross21[:,ind2,ind3],cl_auto2[:,ind2,ind4])[:ell_size, :ell_size]
                    #cov_temp = nmt.gaussian_covariance(ell_coupling,cl_model[:,ind1,ind3],cl_model[:,ind1,ind4],cl_model[:,ind2,ind3],cl_model[:,ind2,ind4])[:ell_size, :ell_size]
                    if not sparse_ell:
                        cov_output[:,ind1,ind2,:,ind3,ind4] = cov_temp
                    else:
                        sparse_diags = []
                        for num in ell_diags:
                            if num==0:
                                sparse_diags.append(np.diag(cov_temp))
                            elif num>0:
                                sparse_diags.append(np.diag(cov_temp[:-num,num:]))
                            else:
                                sparse_diags.append(np.diag(cov_temp[-num:,:num]))
                            #print num
                        #print len(ell_rows)
                        #print len(sparse_diags)
                        temp_sparse = scipy.sparse.diags(sparse_diags, ell_diags, format='coo')
                        sparse_values.append(temp_sparse.data)
                        ell_cols = temp_sparse.col
                        ell_rows = temp_sparse.row
                        full_coord = np.ravel_multi_index((ell_rows, ind1, ind2, ell_cols, ind3, ind4),(ell_size, z_size, z_size, ell_size, z_size, z_size))
                        full_coords = np.unravel_index(full_coord, (ell_size*z_size*z_size,ell_size*z_size*z_size))
                        sparse_row.append(full_coords[0])
                        sparse_col.append(full_coords[1])
    if sparse_ell:
        cov_output = scipy.sparse.coo_matrix((np.array(sparse_values).flatten(), (np.array(sparse_row).flatten(),np.array(sparse_col).flatten())), shape=(ell_size*z_size*z_size,ell_size*z_size*z_size))
    return cov_output

def upper_diag_index(z1,z2):
    #Returns the flattened index for upper diagonal [z1,z2] matrix, where the index is zero at the upper left and populates the columns from left to right and top to bottom.
    ans = (z2*z2 + z2)/2 + z1
    try:
        ans = int(ans)
    except TypeError:
        ans = ans.astype(int)
    return ans

def compute_autopower_cov(ell_coupling, workspace, cl_model, ell_size, sparse_ell = False, ell_diags = [0,1,-1]):
    #Ell_coupling should be NaMaster NmtCovarianceWorkspace object with coupling coefficients calculated.
    #Cl model should have dimensions [ell, z, z].
    #Set sparse_ell to True to use scipy sparse matrix and ignore ell correlations between distant ell-bins.
    #ell_diags sets the diagonals and off-diagonals to include in the sparse approximation of the ell coupling matrix.
    #This algorithm assumes that the upper diagonal of Cl(z,z') is kept (lower part is redundant in auto-power). This means z'>=z.
    try:
        z_size = cl_model.shape[-1]
    except AttributeError:
        z_size = cl_model[0].shape[-1]
    tot_z_size = int(z_size*(z_size+1)/2)
    ell_vect = np.arange(ell_size)
    if not sparse_ell:
        cov_output = np.zeros((ell_size, tot_z_size, ell_size, tot_z_size))
    if sparse_ell:
        sparse_col_small = []
        sparse_row_small = []
        for num in ell_diags:
            if num==0:
                sparse_col_small.append(ell_vect)
                sparse_row_small.append(ell_vect)
            elif num>0:
                sparse_col_small.append(ell_vect[num:])
                sparse_row_small.append(ell_vect[:-num])
            else:
                sparse_col_small.append(ell_vect[:num])
                sparse_row_small.append(ell_vect[-num:])
        sparse_col_small = np.concatenate(sparse_col_small)
        sparse_row_small = np.concatenate(sparse_row_small)
        ell_block_size = sparse_col_small.size
        sparse_col = np.tile(sparse_col_small, tot_z_size*tot_z_size)
        sparse_row = np.tile(sparse_row_small, tot_z_size*tot_z_size)
        sparse_values = np.zeros((sparse_col.size))
    counter=0
    for ind2 in range(z_size):
        for ind1 in range(ind2+1):
            for ind4 in range(z_size):
                for ind3 in range(ind4+1):
                    if isinstance(cl_model, list):
                        cl_auto1 = cl_model[0]
                        cl_cross12 = cl_model[1]
                        cl_cross21 = cl_model[2]
                        cl_auto2 = cl_model[3]
                        cov_temp = nmt.gaussian_covariance(ell_coupling, 0, 0, 0, 0, [cl_auto1[:,ind1,ind3]],[cl_cross12[:,ind1,ind4]],[cl_cross21[:,ind2,ind3]],[cl_auto2[:,ind2,ind4]], workspace, wb=workspace)[:ell_size, :ell_size] 
                        #cov_temp = nmt.gaussian_covariance(ell_coupling,cl_auto1[:,ind1,ind3],cl_cross12[:,ind1,ind4],cl_cross21[:,ind2,ind3],cl_auto2[:,ind2,ind4])[:ell_size, :ell_size]
                    else:
                        cov_temp = nmt.gaussian_covariance(ell_coupling, 0, 0, 0, 0, [cl_model[:,ind1,ind3]],[cl_model[:,ind1,ind4]],[cl_model[:,ind2,ind3]],[cl_model[:,ind2,ind4]], workspace, wb=workspace)[:ell_size, :ell_size]
                        #cov_temp = nmt.gaussian_covariance(ell_coupling,cl_model[:,ind1,ind3],cl_model[:,ind1,ind4],cl_model[:,ind2,ind3],cl_model[:,ind2,ind4])[:ell_size, :ell_size]
                    full_ind1 = upper_diag_index(ind1,ind2)
                    full_ind2 = upper_diag_index(ind3,ind4)
                    if not sparse_ell:
                        cov_output[:,full_ind1,:,full_ind2] = cov_temp
                    else:
                        tot_num = 0
                        for num in ell_diags:
                            if num==0:
                                sparse_values[ell_block_size*counter + tot_num:ell_block_size*counter + tot_num + ell_size] = np.diag(cov_temp)
                            elif num>0:
                                sparse_values[ell_block_size*counter + tot_num:ell_block_size*counter + tot_num + ell_size - num] = np.diag(cov_temp[:-num,num:])
                            else:
                                sparse_values[ell_block_size*counter + tot_num:ell_block_size*counter + tot_num + ell_size + num] = np.diag(cov_temp[-num:,:num])
                            tot_num += ell_size - np.abs(num)
                        sparse_row[ell_block_size*counter:ell_block_size*(counter+1)] = np.ravel_multi_index((sparse_row_small, full_ind1),(ell_size, tot_z_size))
                        sparse_col[ell_block_size*counter:ell_block_size*(counter+1)] = np.ravel_multi_index((sparse_col_small, full_ind2),(ell_size, tot_z_size))
                        counter += 1
    if sparse_ell:
        cov_output = scipy.sparse.coo_matrix((sparse_values,(sparse_row,sparse_col)), shape=(ell_size*tot_z_size,ell_size*tot_z_size))
    return cov_output     

def pcl_cov(ell_ell_double_mask_mixing_matrix, binning_matrix, cl_models, ells, brown_approx = False):
    #1d cl models.
    #Assume cl_model has already been multiplied by beam/pixel transfer function.
    if not brown_approx:
        fact1 = ell_ell_double_mask_mixing_matrix*cl_models[0][:,None]*cl_models[3][None,:]
        fact1 += ell_ell_double_mask_mixing_matrix*cl_models[1][:,None]*cl_models[2][None,:]
    else:
        fact1 = ell_ell_double_mask_mixing_matrix*np.sign(cl_models[0][:,None]*cl_models[3][:,None])*(np.abs(cl_models[0][:,None]*cl_models[3][:,None]))**0.5*np.sign(cl_models[0][None,:]*cl_models[3][None,:])*(np.abs(cl_models[0][None,:]*cl_models[3][None,:]))**0.5
        fact1 += ell_ell_double_mask_mixing_matrix*np.sign(cl_models[1][:,None]*cl_models[2][:,None])*(np.abs(cl_models[1][:,None]*cl_models[2][:,None]))**0.5*np.sign(cl_models[1][None,:]*cl_models[2][None,:])*(np.abs(cl_models[1][None,:]*cl_models[2][None,:]))**0.5
    fact1 /= (2.*ells[None,:]+1.)
    #return np.einsum('ij,lk,jk->il', binning_matrix, binning_matrix, fact1)
    a = np.dot(binning_matrix, fact1)
    return np.dot(a, binning_matrix.T)

def pcl_cov_cross(ell_ell_double_mask_mixing_matrix_a, ell_ell_double_mask_mixing_matrix_b, binning_matrix, cl_models, ells, brown_approx = False):
    #1d cl models.
    #Assume cl_model has already been multiplied by beam/pixel transfer function.
    if not brown_approx:
        fact1 = ell_ell_double_mask_mixing_matrix_a*cl_models[0][:,None]*cl_models[3][None,:]
        fact1 += ell_ell_double_mask_mixing_matrix_b*cl_models[1][:,None]*cl_models[2][None,:]
    else:
        fact1 = ell_ell_double_mask_mixing_matrix_a*np.sign(cl_models[0][:,None]*cl_models[3][:,None])*(np.abs(cl_models[0][:,None]*cl_models[3][:,None]))**0.5*np.sign(cl_models[0][None,:]*cl_models[3][None,:])*(np.abs(cl_models[0][None,:]*cl_models[3][None,:]))**0.5
        fact1 += ell_ell_double_mask_mixing_matrix_b*np.sign(cl_models[1][:,None]*cl_models[2][:,None])*(np.abs(cl_models[1][:,None]*cl_models[2][:,None]))**0.5*np.sign(cl_models[1][None,:]*cl_models[2][None,:])*(np.abs(cl_models[1][None,:]*cl_models[2][None,:]))**0.5
    fact1 /= (2.*ells[None,:]+1.)
    a = np.dot(binning_matrix, fact1)
    return np.dot(a, binning_matrix.T)

def cl_cov_from_pcl_cov(inv_binned_mixing1, inv_binned_mixing2, pcl_cov):
    #print(pcl_cov.shape)
    #print(inv_binned_mixing1.shape)
    #print(inv_binned_mixing2.shape)
    #return np.einsum('ij,jk,lk->il', inv_binned_mixing1, pcl_cov, inv_binned_mixing2)
    a = np.dot(inv_binned_mixing1, pcl_cov)
    return np.dot(a, inv_binned_mixing2.T)
    

def compute_autopower_cov_no_namaster(ell_ell_double_mask_mixing_matrix, binned_mixing_matrix_inv, binning_matrix, cl_model, ell_size, sparse_ell = False, ell_diags = [0,1,-1], brown_approx = False):
    #Cl model should have dimensions [ell, z, z].
    #Set sparse_ell to True to use scipy sparse matrix and ignore ell correlations between distant ell-bins.
    #ell_diags sets the diagonals and off-diagonals to include in the sparse approximation of the ell coupling matrix.
    #This algorithm assumes that the upper diagonal of Cl(z,z') is kept (lower part is redundant in auto-power). This means z'>=z.
    try:
        z_size = cl_model.shape[-1]
    except AttributeError:
        z_size = cl_model[0].shape[-1]
    tot_z_size = int(z_size*(z_size+1)/2)
    ell_vect = np.arange(ell_size)
    all_ells = np.arange(ell_ell_double_mask_mixing_matrix.shape[0])
    if not sparse_ell:
        cov_output = np.zeros((ell_size, tot_z_size, ell_size, tot_z_size))
    if sparse_ell:
        sparse_col_small = []
        sparse_row_small = []
        for num in ell_diags:
            if num==0:
                sparse_col_small.append(ell_vect)
                sparse_row_small.append(ell_vect)
            elif num>0:
                sparse_col_small.append(ell_vect[num:])
                sparse_row_small.append(ell_vect[:-num])
            else:
                sparse_col_small.append(ell_vect[:num])
                sparse_row_small.append(ell_vect[-num:])
        sparse_col_small = np.concatenate(sparse_col_small)
        sparse_row_small = np.concatenate(sparse_row_small)
        ell_block_size = sparse_col_small.size
        sparse_col = np.tile(sparse_col_small, tot_z_size*tot_z_size)
        sparse_row = np.tile(sparse_row_small, tot_z_size*tot_z_size)
        sparse_values = np.zeros((sparse_col.size))
    counter=0
    for ind2 in range(z_size):
        for ind1 in range(ind2+1):
            for ind4 in range(z_size):
                for ind3 in range(ind4+1):
                    if isinstance(cl_model, list):
                        cl_auto1 = cl_model[0]
                        cl_cross12 = cl_model[1]
                        cl_cross21 = cl_model[2]
                        cl_auto2 = cl_model[3]
                        cov_temp = nmt.gaussian_covariance(ell_coupling, 0, 0, 0, 0, [cl_auto1[:,ind1,ind3]],[cl_cross12[:,ind1,ind4]],[cl_cross21[:,ind2,ind3]],[cl_auto2[:,ind2,ind4]], workspace, wb=workspace)[:ell_size, :ell_size] 
                        #cov_temp = nmt.gaussian_covariance(ell_coupling,cl_auto1[:,ind1,ind3],cl_cross12[:,ind1,ind4],cl_cross21[:,ind2,ind3],cl_auto2[:,ind2,ind4])[:ell_size, :ell_size]
                    else:
                        cov_temp = pcl_cov(ell_ell_double_mask_mixing_matrix, binning_matrix, [cl_model[:,ind1,ind3],cl_model[:,ind1,ind4],cl_model[:,ind2,ind3],cl_model[:,ind2,ind4]], all_ells, brown_approx = brown_approx)
                        cov_temp = cl_cov_from_pcl_cov(binned_mixing_matrix_inv[:ell_size,:], binned_mixing_matrix_inv[:ell_size,:], cov_temp)
                        #cov_temp = nmt.gaussian_covariance(ell_coupling,cl_model[:,ind1,ind3],cl_model[:,ind1,ind4],cl_model[:,ind2,ind3],cl_model[:,ind2,ind4])[:ell_size, :ell_size]
                    full_ind1 = upper_diag_index(ind1,ind2)
                    full_ind2 = upper_diag_index(ind3,ind4)
                    if not sparse_ell:
                        cov_output[:,full_ind1,:,full_ind2] = cov_temp
                    else:
                        tot_num = 0
                        for num in ell_diags:
                            if num==0:
                                sparse_values[ell_block_size*counter + tot_num:ell_block_size*counter + tot_num + ell_size] = np.diag(cov_temp)
                            elif num>0:
                                sparse_values[ell_block_size*counter + tot_num:ell_block_size*counter + tot_num + ell_size - num] = np.diag(cov_temp[:-num,num:])
                            else:
                                sparse_values[ell_block_size*counter + tot_num:ell_block_size*counter + tot_num + ell_size + num] = np.diag(cov_temp[-num:,:num])
                            tot_num += ell_size - np.abs(num)
                        sparse_row[ell_block_size*counter:ell_block_size*(counter+1)] = np.ravel_multi_index((sparse_row_small, full_ind1),(ell_size, tot_z_size))
                        sparse_col[ell_block_size*counter:ell_block_size*(counter+1)] = np.ravel_multi_index((sparse_col_small, full_ind2),(ell_size, tot_z_size))
                        counter += 1
    if sparse_ell:
        cov_output = scipy.sparse.coo_matrix((sparse_values,(sparse_row,sparse_col)), shape=(ell_size*tot_z_size,ell_size*tot_z_size))
    return cov_output          

def cov_repeats_no_namaster(ell_ell_double_mask_mixing_matrix_a, ell_ell_double_mask_mixing_matrix_b, binned_mixing_matrix_inv_a, binned_mixing_matrix_inv_b, binning_matrix, cl_models, ell_size, sparse_ell = False, ell_diags = [0,1,-1], brown_approx = False):
    ell_vect = np.arange(ell_size)
    all_ells = np.arange(ell_ell_double_mask_mixing_matrix_a.shape[0])
    z_size = cl_models[0].shape[-1]
    if not sparse_ell:
        cov_output = np.zeros((ell_size, z_size, z_size, ell_size, z_size, z_size))
    if sparse_ell:
        sparse_values = []
        sparse_col = []
        sparse_row = []
    for ind1 in range(z_size):
        for ind2 in range(z_size):
            for ind3 in range(z_size):
                for ind4 in range(z_size):
                    cl_auto1 = cl_models[0]
                    cl_cross12 = cl_models[1]
                    cl_cross21 = cl_models[2]
                    cl_auto2 = cl_models[3]
                    #cov_temp = nmt.gaussian_covariance(ell_coupling, 0, 0, 0, 0, [cl_auto1[:,ind1,ind3]],[cl_cross12[:,ind1,ind4]],[cl_cross21[:,ind2,ind3]],[cl_auto2[:,ind2,ind4]], workspace, wb=workspace)[:ell_size, :ell_size]
                    #cov_temp = nmt.gaussian_covariance(ell_coupling,cl_auto1[:,ind1,ind3],cl_cross12[:,ind1,ind4],cl_cross21[:,ind2,ind3],cl_auto2[:,ind2,ind4])[:ell_size, :ell_size]
                    #cov_temp = nmt.gaussian_covariance(ell_coupling,cl_model[:,ind1,ind3],cl_model[:,ind1,ind4],cl_model[:,ind2,ind3],cl_model[:,ind2,ind4])[:ell_size, :ell_size]
                    cov_temp = pcl_cov_cross(ell_ell_double_mask_mixing_matrix_a, ell_ell_double_mask_mixing_matrix_b, binning_matrix, [cl_auto1[:,ind1,ind3],cl_cross12[:,ind1,ind4],cl_cross21[:,ind2,ind3],cl_auto2[:,ind2,ind4]], all_ells, brown_approx = brown_approx)
                    cov_temp = cl_cov_from_pcl_cov(binned_mixing_matrix_inv_a[:ell_size,:], binned_mixing_matrix_inv_b[:ell_size,:], cov_temp)
                    if not sparse_ell:
                        cov_output[:,ind1,ind2,:,ind3,ind4] = cov_temp
                    else:
                        sparse_diags = []
                        for num in ell_diags:
                            if num==0:
                                sparse_diags.append(np.diag(cov_temp))
                            elif num>0:
                                sparse_diags.append(np.diag(cov_temp[:-num,num:]))
                            else:
                                sparse_diags.append(np.diag(cov_temp[-num:,:num]))
                            #print num
                        #print len(ell_rows)
                        #print len(sparse_diags)
                        temp_sparse = scipy.sparse.diags(sparse_diags, ell_diags, format='coo')
                        sparse_values.append(temp_sparse.data)
                        ell_cols = temp_sparse.col
                        ell_rows = temp_sparse.row
                        full_coord = np.ravel_multi_index((ell_rows, ind1, ind2, ell_cols, ind3, ind4),(ell_size, z_size, z_size, ell_size, z_size, z_size))
                        full_coords = np.unravel_index(full_coord, (ell_size*z_size*z_size,ell_size*z_size*z_size))
                        sparse_row.append(full_coords[0])
                        sparse_col.append(full_coords[1])
    if sparse_ell:
        cov_output = scipy.sparse.coo_matrix((np.array(sparse_values).flatten(), (np.array(sparse_row).flatten(),np.array(sparse_col).flatten())), shape=(ell_size*z_size*z_size,ell_size*z_size*z_size))
    return cov_output

    

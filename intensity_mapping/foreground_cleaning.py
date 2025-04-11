import numpy as np

def linear_comb(input_map, templates, weight, coefs=None):
    """Find the linear combination that minimizes variance in input_map
    given list of templates and a weight
    """
    temp_mat = np.stack(templates)
    temp_means = np.dot(temp_mat, weight) / np.sum(weight)
    temp_mat -= temp_means[:, None]
    #print "means:", temp_means

    input_nomean = input_map - np.dot(input_map, weight) / np.sum(weight)

    if coefs is None:
        proj = np.dot(temp_mat, weight * input_nomean)
        weighted_temp = weight[None, :] * temp_mat
        norm = np.linalg.pinv(np.dot(temp_mat, weighted_temp.T))
        coefs = np.dot(norm, proj)

    cleaned = input_nomean - np.dot(coefs, temp_mat)
    return coefs, cleaned

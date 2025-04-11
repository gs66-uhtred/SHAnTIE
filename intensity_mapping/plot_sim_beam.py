import numpy as np
import matplotlib.pyplot as plt

def directivity_to_power(data):
    return 10**(data/10.)

def theta_phi_to_x_y(theta, phi, output = 'degrees'):
    #theta and phi in degrees.
    theta_rad = theta*np.pi/180.
    phi_rad = phi*np.pi/180.
    if output == 'degrees':
        x = theta*np.cos(phi_rad)
        y = theta*np.sin(phi_rad)
    else:
        x = theta_rad*np.cos(phi_rad)
        y = theta_rad*np.sin(phi_rad)
    return x,y

def scatter_plot_helper(theta, phi, max_deg, power, fname = 'test', size = 4, title = None, cmap = 'viridis'):
    #theta and phi in degrees.
    edge_bool = theta <= max_deg
    x,y = theta_phi_to_x_y(theta[edge_bool], phi[edge_bool])
    plt.scatter(x, y, c=power[edge_bool], s=size, alpha=1.,
            cmap=cmap)
    plt.colorbar()
    if type(title) != type(None):
        plt.title(title)
    plt.show()
    plt.savefig(fname)

def scatter_plot_split_pos_neg(theta, phi, max_deg, power, fname = 'test', size = 4, title = None):
    #theta and phi in degrees.
    edge_bool0 = theta[0] <= max_deg
    x0,y0 = theta_phi_to_x_y(theta[0][edge_bool0], phi[0][edge_bool0])
    edge_bool1 = theta[1] <= max_deg
    x1,y1 = theta_phi_to_x_y(theta[1][edge_bool1], phi[1][edge_bool1])
    plt.scatter(x0, y0, c=power[0][edge_bool0], s=size, alpha=1.,
            cmap='Greens')
    plt.colorbar()
    plt.scatter(x1, y1, c=power[1][edge_bool1], s=size, alpha=1., cmap='Purples')
    plt.colorbar()
    if type(title) != type(None):
        plt.title(title)
    plt.show()
    plt.savefig(fname)

def stokes_formula(co_power1, co_phase1, x_power1, x_phase1, co_power2, co_phase2, x_power2, x_phase2, stokes_param = 'I', axes_rot = True, co_is_x = True, pols_corr = 0.):
    #If axes_rot, assume co_power1 and x_power2 are the same direction. Otherwise, co_power1 and co_power2 are the same direction.
    #If co_is_x, treat co_polar direction (of 1) as the x direction for computing Stokes' parameters. Otherwise, treat x_polar as x.
    #Assume power from dipole 1 and dipole 2 have correlation pols_corr. Random uncorrelated phases mean pols_corr = 0.
    if axes_rot:
        co_power2, x_power2 = x_power2, co_power2
        co_phase2, x_phase2 = x_phase2, co_phase2
    if not co_is_x:
        co_power2, x_power2 = x_power2, co_power2
        co_phase2, x_phase2 = x_phase2, co_phase2
        co_power1, x_power1 = x_power1, co_power1
        co_phase1, x_phase1 = x_phase1, co_phase1
    co_corr = np.cos((co_phase1 - co_phase2)*np.pi/180.)*pols_corr
    x_corr = np.cos((x_phase1 - x_phase2)*np.pi/180.)*pols_corr
    if stokes_param == 'I':
        answer = co_power1 + co_power2 + 2.*co_corr*(co_power1*co_power2)**0.5
        answer += x_power1 + x_power2 + 2.*x_corr*(x_power1*x_power2)**0.5
    if stokes_param == 'Q':
        answer = co_power1 + co_power2 + 2.*co_corr*(co_power1*co_power2)**0.5
        answer -= x_power1 + x_power2 + 2.*x_corr*(x_power1*x_power2)**0.5
    if stokes_param == 'U':
        xcorr11 = np.cos((co_phase1 - x_phase1)*np.pi/180.)
        xcorr22 = np.cos((co_phase2 - x_phase2)*np.pi/180.)
        xcorr12 = np.cos((co_phase1 - x_phase2)*np.pi/180.)
        xcorr21 = np.cos((co_phase2 - x_phase1)*np.pi/180.)
        answer = 2.*xcorr11*(co_power1*x_power1)**0.5 + 2.*pols_corr*xcorr12*(co_power1*x_power2)**0.5
        answer += 2.*pols_corr*xcorr21*(co_power2*x_power1)**0.5 + 2.*xcorr22*(co_power2*x_power2)**0.5
    if stokes_param == 'V':
        xcorr11 = -np.sin((co_phase1 - x_phase1)*np.pi/180.)
        xcorr22 = -np.sin((co_phase2 - x_phase2)*np.pi/180.)
        xcorr12 = -np.sin((co_phase1 - x_phase2)*np.pi/180.)
        xcorr21 = -np.sin((co_phase2 - x_phase1)*np.pi/180.)
        answer = 2.*xcorr11*(co_power1*x_power1)**0.5 + 2.*pols_corr*xcorr12*(co_power1*x_power2)**0.5
        answer += 2.*pols_corr*xcorr21*(co_power2*x_power1)**0.5 + 2.*xcorr22*(co_power2*x_power2)**0.5
    return answer

def scatter_plot_data(pol1_data, pol2_data, max_deg=0.5, axes_rot = True, stokes_param = 'I', pols_corr = 0., co_is_x = True, fname = 'test', size = 4, logplot=False):
    #Assumed format, column:
    #0. Theta, 1. Phi, 2. total dir, 3. x_dir, 4. x_phase, 5. co_dir, 6. co_phase.
    #Assume power from dipole 1 and dipole 2 have correlation pols_corr. Random uncorrelated phases mean pols_corr = 0.
    power_data = stokes_formula(directivity_to_power(pol1_data[:,5]), pol1_data[:,6], directivity_to_power(pol1_data[:,3]), pol1_data[:,4], directivity_to_power(pol2_data[:,5]), pol2_data[:,6], directivity_to_power(pol2_data[:,3]), pol2_data[:,4], stokes_param = stokes_param, axes_rot = axes_rot, co_is_x = co_is_x, pols_corr = 0.)
    if not logplot:
        scatter_plot_helper(pol1_data[:,0], pol1_data[:,1], max_deg, power_data, fname = fname, size = size, title = stokes_param)
    else:
        neg_bool = power_data < 0.
        if np.sum(neg_bool) == 0:
            scatter_plot_helper(pol1_data[:,0], pol1_data[:,1], max_deg, np.log10(power_data), fname = fname, size = size, title = stokes_param, cmap = 'Greens')
        else:
            pos_bool = np.logical_not(neg_bool)
            theta = [pol1_data[:,0][pos_bool],pol1_data[:,0][neg_bool]]
            phi = [pol1_data[:,1][pos_bool],pol1_data[:,1][neg_bool]]
            power_data = [np.log10(power_data[pos_bool]),np.log10(-power_data[neg_bool])]
            scatter_plot_split_pos_neg(theta, phi, max_deg, power_data, fname = fname, size = size, title = stokes_param)
    return np.array([pol1_data[:,0], pol1_data[:,1], power_data])

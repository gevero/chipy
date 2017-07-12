# libraries
import numpy as np
import scipy as sp
from numba import jit

# constants
from scipy.constants import speed_of_light as c
from scipy.constants import mu_0
z_0 = sp.constants.value('characteristic impedance of vacuum')


@jit(nopython=True)
def f_k_cos(omega,k_0,theta_0,v_d,v_eps,v_csi):
    '''
    calculates the k_n,k_n_norm and generalized fresnel angles theta_n for all
    the layers, plus the incident medium and the substrate

    Inputs
    ----------
    'omega' = light frequency
    'k_0' = incident medium wavevector
    'thetha_0' = incident angle in radians
    'v_d' = multilayer thicknesses, inc and sub are set to 0.0
    'v_eps' = multilayer dielectric constants, inc and sub must be real
    'v_csi' = multilayer chirality parameters, inc and sub must be set to 0.0

    Outputs
    -------
    'v_k_p,v_k_m'= wavevector moduli for left and righ polarization
    'v_kn_p,v_kn_m'= wavevector normal component for left and righ polarization
    'v_cos_p,v_cos_m'= cosines of fresnel angles for left and right polarization
    '''
    # wavevector moduli
    v_k_p = omega*(np.sqrt(v_eps / (c**2) + (mu_0**2)*(v_csi**2)) + mu_0 * v_csi + 0j)
    v_k_m = omega*(np.sqrt(v_eps / (c**2) + (mu_0**2)*(v_csi**2)) - mu_0 * v_csi + 0j)

    # normal component
    k_p = k_0*np.sin(theta_0)  # wavevector paralell component (equal everywhere)
    v_kn_p = np.sqrt(v_k_p**2 - k_p**2)
    v_kn_m = np.sqrt(v_k_m**2 - k_p**2)

    # angle cosines
    v_cos_p = v_kn_p / v_k_p
    v_cos_m = v_kn_m / v_k_m

    return v_k_p, v_k_m, v_kn_p, v_kn_m, v_cos_p, v_cos_m


@jit(nopython=True)
def f_M_P(omega,k_0,theta_0,v_d,v_eps,v_csi):
    '''
    calculates the transfer interface matrix (m) and the transfer phase matrix P for
    each interface and each layer

    Inputs
    ----------
    'omega' = light frequency
    'k_0' = incident medium wavevector
    'thetha_0' = incident angle in radians
    'v_d' = multilayer thicknesses, inc and sub are set to 0.0
    'v_eps' = multilayer dielectric constants, inc and sub must be real
    'v_csi' = multilayer chirality parameters, inc and sub must be set to 0.0

    Outputs
    -------
    'm_M'= interface transfer matrix
    'm_P'= phase transfer matrix
    '''

    # intrinsic wave impedance for each layer
    v_eta = 1.0 / np.sqrt(v_eps / (z_0**2) + v_csi**2 + 0j)

    # wavevectors and cosines for each layer
    v_k_p,v_k_m,v_kn_p,v_kn_m,v_cos_p,v_cos_m = f_k_cos(omega,k_0,theta_0,v_d,v_eps,v_csi)

    # initializing M and P storage
    n_layers = len(v_eps)-2
    m_M = np.zeros((4,4,n_layers+1),dtype=np.complex128)
    m_P = np.zeros((4,4,n_layers+1),dtype=np.complex128)

    # building M
    v_eta_r = v_eta[0:-1] / v_eta[1:]
    m_M[0,0,:] = ((v_eta_r + 1.0) / 4.0) * (1 + v_cos_p[1:] / v_cos_p[0:-1])  # M_T
    m_M[0,1,:] = ((v_eta_r - 1.0) / 4.0) * (1 - v_cos_m[1:] / v_cos_p[0:-1])
    m_M[1,0,:] = ((v_eta_r - 1.0) / 4.0) * (1 - v_cos_p[1:] / v_cos_m[0:-1])
    m_M[1,1,:] = ((v_eta_r + 1.0) / 4.0) * (1 + v_cos_m[1:] / v_cos_m[0:-1])
    m_M[0,2,:] = ((v_eta_r + 1.0) / 4.0) * (1 - v_cos_p[1:] / v_cos_p[0:-1])  # M_R
    m_M[0,3,:] = ((v_eta_r - 1.0) / 4.0) * (1 + v_cos_m[1:] / v_cos_p[0:-1])
    m_M[1,2,:] = ((v_eta_r - 1.0) / 4.0) * (1 + v_cos_p[1:] / v_cos_m[0:-1])
    m_M[1,3,:] = ((v_eta_r + 1.0) / 4.0) * (1 - v_cos_m[1:] / v_cos_m[0:-1])
    m_M[2:4,2:4,:] = m_M[0:2,0:2,:]  # M_T
    m_M[2:4,0:2,:] = m_M[0:2,2:4,:]  # M_R

    # building P
    m_P[0,0,:] = np.exp(-1j * v_kn_p[0:-1] * v_d[0:-1])
    m_P[1,1,:] = np.exp(-1j * v_kn_m[0:-1] * v_d[0:-1])
    m_P[2,2,:] = np.exp(1j * v_kn_p[0:-1] * v_d[0:-1])
    m_P[3,3,:] = np.exp(1j * v_kn_m[0:-1] * v_d[0:-1])

    return m_M, m_P


@jit(nopython=True)
def f_R_T(wl,theta_0,v_d,v_eps,v_csi):
    '''
    calculates reflection and transmission matrix of a chiral multilayer

    Inputs
    ----------
    'wl' = incident wavelength in nm
    'theta_0' = incident angle in radians
    'v_d' = multilayer thicknesses in nm, inc and sub are set to 0.0
    'v_eps' = multilayer dielectric constants, inc and sub must be real
    'v_csi' = multilayer chirality parameters, inc and sub must be set to 0.0

    Outputs
    -------
    'm_T_rl' = chiral multilayer reflection matrix in circular polarization basis
    'm_R_rl' = chiral multilayer transmission matrix  in circular polarization basis
    'm_T_ps' = chiral multilayer reflection matrix in linear polarization basis
    'm_R_ps' = chiral multilayer transmission matrix  in linear polarization basis
    '''

    # bringing everything to MKS
    wl = wl * 1e-9
    v_d = v_d * 1e-9

    # light frequency and incident wavevector modulus
    omega = c * 2.0 * np.pi / wl
    k_0 = 2.0 * np.pi * np.sqrt(v_eps[0]) / wl

    # interface and phase transfer matrix
    m_M, m_P = f_M_P(omega,k_0,theta_0,v_d,v_eps,v_csi)

    # looping to get m_S
    n_layers = len(v_eps)-2
    m_S = np.copy(m_M[:,:,0])
    for n in range(1,(n_layers+1)):
        m_S = np.dot(m_S,np.dot(m_P[:,:,n],m_M[:,:,n]))

    # R and T
    m_C = np.zeros((2,2),dtype=np.complex128)  # change of base matrix
    m_C[0,0] = 1.0
    m_C[0,1] = 1.0*1j
    m_C[1,0] = 1.0
    m_C[1,1] = -1.0*1j
    # m_C = np.array([[1.0, 1j],[1.0, -1j]])  # change of base matrix
    m_C = (1.0/np.sqrt(2.0)) * m_C
    m_R_rl = np.dot(m_S[2:4,0:2],np.linalg.inv(m_S[0:2,0:2]))  # circular polarization
    m_T_rl = np.linalg.inv(m_S[0:2,0:2])
    m_R_ps = np.dot(np.linalg.inv(m_C),np.dot(m_R_rl,m_C))  # linear polarization
    m_T_ps = np.dot(np.linalg.inv(m_C),np.dot(m_T_rl,m_C))

    return m_T_rl, m_R_rl, m_T_ps, m_R_ps, m_M, m_P, m_S

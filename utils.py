# libraries
import numpy as np
import scipy as sp
from numba import jit

# constants
from scipy.constants import speed_of_light as c
from scipy.constants import mu_0
from scipy.constants import hbar
from scipy.constants import elementary_charge
z_0 = sp.constants.value('characteristic impedance of vacuum')
hbar_ev = hbar/elementary_charge


@jit(nopython=True)
def f_slab(wl,d,eps1,eps,csi):
    '''
    calculates normal transmission and reflection matrix for a chiral slab in a symmetric dielectric
    medium

    Inputs
    ----------
    'wl' = incident wavelength in nm
    'd' = chiral layer thickness
    'eps1,eps' = dielectric constant of the surrounding medium and of the chiral slab
    'csi' = chirality parameter


    Outputs
    -------
    'm_R_ps' = chiral reflection matrix in linear polarization basis
    'm_T_ps' = chiral transmission matrix  in linear polarization basis
    '''

    # bringing everything to MKS
    wl = wl * 1e-9
    d = d * 1e-9

    # light frequency and incident wavevector modulus
    omega = c * 2.0 * np.pi / wl
    k_i = 2.0 * np.pi * np.sqrt(eps1+0j) / wl

    # wavevectors inside the slab
    k_1 = omega*(np.sqrt(eps / (c**2) + (mu_0**2)*(csi**2)) + mu_0 * csi + 0j)
    k_2 = omega*(np.sqrt(eps / (c**2) + (mu_0**2)*(csi**2)) - mu_0 * csi + 0j)

    # phase deltas
    delta_1 = k_1*d
    delta_2 = k_2*d
    delta_i = k_i*d

    # g factor
    g = np.sqrt((z_0**2)*(csi**2)/eps1 + eps/eps1 + 0j)
    g_div1 = ((1+g)/(1-g))

    # auxiliary quantities for r and t
    g_div1 = ((1+g)/(1-g))
    g_div2 = 2*g/((1-g)**2)
    exp_12 = np.exp(1j*(delta_1+delta_2))
    exp_1i = np.exp(1j*(delta_1-delta_i))
    exp_2i = np.exp(1j*(delta_2-delta_i))

    # reflection and transmission matriz
    m_r = np.zeros((2,2),dtype=np.complex128)
    m_t = np.zeros_like(m_r)
    m_r[0,0] = g_div1*(1-exp_12)/(g_div1**2-exp_12)
    m_r[1,1] = m_r[0,0]
    m_t[0,0] = np.exp(1j*k_i*d)*g_div2*(exp_1i+exp_2i)/(g_div1**2-exp_12)
    m_t[1,1] = m_t[0,0]
    m_t[0,1] = m_t[0,0]*np.tan((delta_2-delta_1)/2.0)
    m_t[1,0] = -m_t[0,1]

    return m_r, m_t


@jit(nopython=True)
def f_chiral(wl,gamma_chi,beta_chi,wl_trans,cap_gamma,eps_b):
    '''
    calculates the chiral dielectric function of a pasteur material with a single transition

    Inputs
    ----------
    'wl' = incident wavelength in nm
    'gamma_chi' = magnitude of absorptive properties in eV
    'beta_chi' = magnitude of chiral properties in eV
    'wl_trans' = wavelength of the electronic transition in nm
    'cap_gamma' = transition damping
    'eps_b' = background optical constant

    Outputs
    -------
    'eps' = optical constant
    'kappa' = pasteur parameter
    '''

    # bringing everything to MKS
    omega = 1240.0/wl  # in eV
    omega_0 = 1240.0/wl_trans  # in eV

    # epsilon
    rho = 1.0/((omega-omega_0)-1j*cap_gamma) - 1.0/((omega+omega_0)-1j*cap_gamma)
    rho_kappa = 1.0/((omega-omega_0)-1j*cap_gamma) + 1.0/((omega+omega_0)-1j*cap_gamma)
    eps = eps_b - gamma_chi*rho
    kappa = beta_chi*rho

    return eps,kappa,rho


@jit(nopython=True)
def f_chiral_xi(wl,gamma_chi,beta_chi,wl_trans,cap_gamma,eps_b):
    '''
    calculates the chiral dielectric function of a pasteur material with a single transition

    Inputs
    ----------
    'wl' = incident wavelength in nm
    'gamma_chi' = magnitude of absorptive properties in eV
    'beta_chi' = magnitude of chiral properties in eV
    'wl_trans' = wavelength of the electronic transition in nm
    'cap_gamma' = transition damping
    'eps_b' = background optical constant

    Outputs
    -------
    'eps' = optical constant
    'kappa' = pasteur parameter
    '''

    # bringing everything to MKS
    omega = 1240.0/wl  # in eV
    omega_0 = 1240.0/wl_trans  # in eV

    # epsilon
    rho_eps = 1.0/((omega-omega_0)+1j*cap_gamma) - 1.0/((omega+omega_0)+1j*cap_gamma)
    rho = 1.0/((omega+omega_0)+1j*cap_gamma) + 1.0/((omega-omega_0)+1j*cap_gamma)
    eps_chi = eps_b - gamma_chi*rho_eps
    xi_chi = beta_chi*rho

    return eps_chi,xi_chi,rho


@jit(nopython=True)
def f_R(m_r,v_jones):
    '''
    calculates reflection starting from an arbitrary jones vector

    Inputs
    ----------
    'm_r' = reflection matrix
    'v_jones' = jones vector

    Outputs
    -------
    'R' = reflectance
    '''

    # jones vector components
    a = v_jones[0]
    b = v_jones[1]

    # reflection matrix components
    r_pp = m_r[0,0]
    r_ps = m_r[0,1]
    r_sp = m_r[1,0]
    r_ss = m_r[1,1]

    R = (np.abs(a)**2 * (np.abs(r_pp)**2 + np.abs(r_sp)**2) +
         np.abs(b)**2 * (np.abs(r_ps)**2 + np.abs(r_ss)**2) +
         # 2.0*np.real(a*np.conj(b)*(r_pp*np.conj(r_ps)+r_sp*np.conj(r_ss))))/(np.abs(a)**2 + np.abs(b)**2)
         2.0*(a*np.conj(b)*(r_pp*np.conj(r_ps)+r_sp*np.conj(r_ss))).real)/(np.abs(a)**2 + np.abs(b)**2)

    return R


# @jit(nopython=True)
def f_T(m_t,v_jones,theta_0,n_inc,n_sub):
    '''
    calculates transmission starting from an arbitrary jones vector

    Inputs
    ----------
    'm_t' = reflection matrix
    'v_jones' = jones vector
    'theta_0' = incident angle
    'n_inc' = incident medium ref index
    'n_sub' = substrate ref index

    Outputs
    -------
    'T' = transmittance
    '''

    # jones vector components
    a = v_jones[0]
    b = v_jones[1]

    # reflection matrix components
    t_pp = m_t[0,0]
    t_ps = m_t[0,1]
    t_sp = m_t[1,0]
    t_ss = m_t[1,1],

    # transmission
    if np.abs(np.sin(theta_0)*n_inc/n_sub) > 1.0:
        T = 0
    else:
        theta_s = np.arcsin(np.sin(theta_0)*n_inc/n_sub)  # exit angle and normalization factor
        # norm = (np.real(n_sub*np.conj(np.cos(theta_s))) /
        #         np.real(n_inc*np.conj(np.cos(theta_0))))
        norm = ((n_sub*np.conj(np.cos(theta_s))).real /
                (n_inc*np.conj(np.cos(theta_0))).real)

        T = norm * (np.abs(a)**2 * (np.abs(t_pp)**2 + np.abs(t_sp)**2) +
                    np.abs(b)**2 * (np.abs(t_ps)**2 + np.abs(t_ss)**2) +
                    # 2.0*np.real(a*np.conj(b)*(t_pp*np.conj(t_ps)+t_sp*np.conj(t_ss))))/(np.abs(a)**2 + np.abs(b)**2)
                    2.0*(a*np.conj(b)*(t_pp*np.conj(t_ps)+t_sp*np.conj(t_ss))).real)/(np.abs(a)**2 + np.abs(b)**2)

    return T

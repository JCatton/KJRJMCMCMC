import scipy.special as sp
import numpy as np
# import mpmath as mp

# def ellippi(n, m):
#   return sp.elliprf(                                                                                                                                                                                        
#         0., 1. - m, 1.) + (n / 3.) * sp.elliprj(0., 1. - m, 1., 1. - n)

import numpy as np

def ellpic_bulirsch(n, k):
    """
    Computes the complete elliptic integral of the third kind using Bulirsch's method.
    
    Parameters:
        n : float or ndarray
            Parameter `n` in the elliptic integral.
        k : float or ndarray
            Modulus `k` in the elliptic integral.
    
    Returns:
        float or ndarray
            The value of the complete elliptic integral of the third kind.
    """
    kc = np.sqrt(1.0 - k**2)
    p = n + 1.0

    # Check for negative p values
    if np.any(p < 0.0):
        raise ValueError("Negative p encountered.")

    # Initialize variables
    m0 = 1.0
    c = 1.0
    p = np.sqrt(p)
    d = 1.0 / p
    e = kc

    tol = 1e-8
    niter = 0
    max_iter = 20

    while niter < max_iter:
        f = c
        c = d / p + c
        g = e / p
        d = 2.0 * (f * g + d)
        p = g + p
        g = m0
        m0 = kc + m0

        # Check convergence
        if np.max(np.abs(1.0 - kc / g)) > tol:
            kc = 2.0 * np.sqrt(e)
            e = kc * m0
        else:
            return 0.5 * np.pi * (c * m0 + d) / (m0 * (m0 + p))
        
        niter += 1

    # If max iterations reached without convergence
    return 0.5 * np.pi * (c * m0 + d) / (m0 * (m0 + p))


def ellec(k):
    """
    Computes the complete elliptic integral of the second kind using Hastings' approximation.
    """
    m1 = 1.0 - k**2
    a1, a2, a3, a4 = 0.44325141463, 0.06260601220, 0.04757383546, 0.01736506451
    b1, b2, b3, b4 = 0.24998368310, 0.09200180037, 0.04069697526, 0.00526449639
    ee1 = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * np.log(1.0 / m1)
    return ee1 + ee2

def ellk(k):
    """
    Computes the complete elliptic integral of the first kind using Hastings' approximation.
    """
    m1 = 1.0 - k**2
    a0, a1, a2, a3, a4 = 1.38629436112, 0.09666344259, 0.03590092383, 0.03742563713, 0.01451196212
    b0, b1, b2, b3, b4 = 0.5, 0.12498593597, 0.06880248576, 0.03328355346, 0.00441787012
    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * np.log(m1)
    return ek1 - ek2

K = sp.ellipk
E = sp.ellipe

Pi = ellpic_bulirsch

def limb_darkening_flux(p: float, z: np.ndarray, third_axis, lambda_e: np.ndarray, limb_coeff_1:float, limb_coeff_2:float):
    """
    """

    c_2 = limb_coeff_1 + 2*limb_coeff_2
    c_4 = -limb_coeff_2

    big_ohm = c_2/(6) + c_4/(8)

    lambda_d = np.zeros_like(z)
    eta_d = np.zeros_like(z)

    case_1 = np.where((p > 0) & (z > 1 + p) | (third_axis < 0))
    case_2 = np.where((third_axis > 0) & (p > 0) & (z > 1/2 + abs(p - 1/2)) & (z < 1 + p))
    case_3 = np.where((third_axis > 0) & (p > 0) & (p < 1 / 2) & (z > p) & (z < 1 - p))
    case_4 = np.where((third_axis > 0) & (p > 0) & (p < 1 / 2) & (z == 1 - p))
    case_5 = np.where((third_axis > 0) & (p > 0) & (p < 1 / 2) & (z == p))
    case_6 = np.where((third_axis > 0) & (p == 1 / 2) & (z == 1 / 2))
    case_7 = np.where((third_axis > 0) & (p > 1 / 2) & (z == p))
    case_8 = np.where((third_axis > 0) & (p > 1 / 2) & (abs(1 - p) < z) & (z < p))
    case_9 = np.where((third_axis > 0) & (p > 0) & (p<1)& (z > 0) & (z < 1 / 2 - abs(p - 1 / 2)))
    case_10 = np.where((third_axis > 0) & (p > 0) & (z == 0))
    case_11 = np.where((third_axis > 0) & (p > 1) & (z > 0) & (z < p - 1))

    lambda_d[case_1] = 0
    eta_d[case_1] = 0

    lambda_d[case_2] = lambda_1(p,z[case_2])
    eta_d[case_2] = eta_1(p,z[case_2])


    lambda_d[case_3] = lambda_2(p,z[case_3])

    lambda_d[case_4] = lambda_5(p,z[case_4])

    lambda_d[case_5] = lambda_4(p,z[case_5])


    lambda_d[case_6] = 1/3 - (4/9) * np.pi


    lambda_d[case_7] = lambda_3(p,z[case_7])


    lambda_d[case_8] = lambda_1(p,z[case_8])

    lambda_d[case_9] = lambda_2(p,z[case_9])
    lambda_d[case_10] = lambda_6(p,z[case_10])
    lambda_d[case_11] = 1

    

    eta_d[case_3] = eta_2(p,z[case_3])

    eta_d[case_4] = eta_2(p,z[case_4])

    eta_d[case_5] = eta_2(p,z[case_5])

    eta_d[case_6] = 3/32

    eta_d[case_7] = eta_1(p,z[case_7])

    eta_d[case_8] = eta_1(p,z[case_8])

    eta_d[case_9] = eta_2(p,z[case_9])

    eta_d[case_10] = eta_2(p,z[case_10])  

    eta_d[case_11] = 1

    correction_term = ( (1-c_2)*lambda_e + c_2*(lambda_d + (2/3) * heavyside(p-z)) - c_4*eta_d)/(4*big_ohm)

    # correction_term[case_1] = 1
    # # correction_term[case_2] = 2 
    # correction_term[case_3] = 3
    # correction_term[case_4] = 4
    # correction_term[case_5] = 5
    # correction_term[case_6] = 6
    # correction_term[case_7] = 7
    # correction_term[case_8] = 8
    # correction_term[case_9] = 9
    # correction_term[case_10] = 10
    # correction_term[case_11] = 11



    return correction_term

def heavyside(x):
    return np.where(x >= 0, 1, 0)
    

def lambda_1(p,z):
    a = a_calculator(p,z)
    b = b_calculator(p,z)
    q = q_calculator(p,z)
    k = k_calculator(p,z,a)

    part_1 = ( (1-b)*(2*b + a - 3)  -  3*q*(b-2) ) * K(k)

    part_2 = 4*p*z*(z*z + 7*p*p -4)*E(k)

    part_3 = -3*(q/a)*Pi((a-1)/a, k)

    multiplication_coefficient = 1/(9*np.pi*np.sqrt(p*z))

    l_1 = multiplication_coefficient*(part_1 + part_2 + part_3)

    return l_1

def lambda_2(p,z):
    a = a_calculator(p,z)
    b = b_calculator(p,z)
    q = q_calculator(p,z)
    k = k_calculator(p,z,a)

    part_1 = ( 1 - 5*z*z  + p*p + q*q) * K(1/k)

    part_2 = (1-a) * (z*z+ 7*p*p - 4) * E(1/k)

    part_3 = -3*(q/a)*Pi((a-b)/a, 1/k)

    multiplication_coefficient = 2/(9*np.pi*np.sqrt(1-a))

    l_2 = multiplication_coefficient*(part_1 + part_2 + part_3)
    return l_2

def lambda_3(p,z):
    a = a_calculator(p,z)
    k = k_calculator(p,z,a)

    part_1 = 1/3
    part_2 = (16*p/9*np.pi)*(2*p*p - 1) * E(1/(2*k))
    part_3 = -(1-4*p*p)*(3-8*p*p)*K(1/(2*k))/(9*p*np.pi)

    l_3 = part_1 + part_2 + part_3

    return l_3

def lambda_4(p,z):
    a = a_calculator(p,z)
    k = k_calculator(p,z,a)

    part_1 = 1/3
    part_2 = 4*(2*p*p-1)*E(2*k)
    part_3 = (1-4*p*p)*K(2*k)


    multiplication_coefficient = 2/(9*np.pi)

    l_4 = part_1 + multiplication_coefficient*(part_2 + part_3)

    return l_4

def lambda_5(p,z):
    part_1 = (2/(3*np.pi)) * np.arccos(1-2*p)

    part_2 = (4/(9*np.pi)) * (3 + 2*p - 8*p*p)

    l_5 = part_1 + part_2

    return l_5

def lambda_6(p,z):

    l_6 = -(2/3)*(1-p*p)**(3/2)

    return l_6

def eta_2(p,z):
    e_2 = (p*p/2)*(p*p+2*z*z)

    return e_2

def eta_1(p,z):
    a = a_calculator(p,z)
    b = b_calculator(p,z)
    multiplication_coefficient = 1/(2*np.pi)
    part_1 = np.arccos((1-p*p+z*z)/(2*z))

    part_2 = 2*eta_2(p,z)*np.arccos((p*p+z*z-1)/(2*p*z))

    part_3 = -(1/4)*(1+5*p*p+z*z)*np.sqrt((1-a)*(b-1))

    e_1 = multiplication_coefficient*(part_1 + part_2 + part_3)

    return e_1




def b_calculator(p,z):
    return (p+z)*(p+z)

def a_calculator(p,z):
    return (p-z)*(p-z)

def q_calculator(p,z):
    return p*p - z*z

def k_calculator(p,z,a):
    return np.sqrt((1-a)/(4*z*p))



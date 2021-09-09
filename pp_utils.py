""" pp_utils.py - This file will implement several functions for applying the standard
ATLAS preprocessing sets to a set of jets. All of these functions will take in
equal length vectors giving the eta, phi, and some energy or pT weight of the jet 
constituents.

Authors: Kevin Greif
9/8/21
python3
"""

import numpy as np

def flip_planes(phi):
    """ flip_planes - This function will fix the discontinuity in phi so that jets which
    happen to fall on this boundary will be pasted back together.

    Arguments:
        phi (array) - The phi values

    Returns:
        (array) - The correct phi values
    """
    
    if np.max(phi) - np.min(phi) > np.pi:
        phi = np.where(phi > 0, phi - np.pi, phi + np.pi)

    return phi


def shift(angles, weights):
    """ shift - This function will shift the values in angles
    such that when weighted by the weights they have zero mean.

    Arguments:
        angles (array) - The values to shift
        weights (array) - The weights to use for shifting

    Returns:
        (array) - The shifted angle values
    """

    return angles - (np.dot(angles, weights) / np.sum(weights))


def rotate(eta, phi, weights):
    """ rotate - This function will apply a rotation in eta, phi
    plane such that the first principle component of the radiation
    described by the weights vector points in the negative phi direction.

    Arguments:
        eta (array) - The eta values
        phi (array) - The phi values
        weights (array) - The weights representing pT or energy

    Returns
        (array, array) - The rotated values for eta and phi
    """

    # Calculate covariance matrix
    w_tot = np.sum(weights)
    mu_eta = np.dot(eta, weights) / w_tot
    mu_phi = np.dot(phi, weights) / w_tot
    mu2_eta = np.dot(eta*eta, weights) / w_tot
    mu2_phi = np.dot(phi*phi, weights) / w_tot
    mu_eta_phi = np.dot(eta*phi, weights) / w_tot

    # Matrix elements calculated here
    sig2_eta = mu2_eta - mu_eta * mu_eta
    sig2_phi = mu2_phi - mu_phi * mu_phi
    sig_eta_phi = mu_eta_phi - mu_eta * mu_phi

    # Compute eigenvalues
    lam_neg = 0.5 * (sig2_eta + sig2_phi - np.sqrt((sig2_eta - sig2_phi) * (sig2_eta - sig2_phi) + 4 * sig_eta_phi * sig_eta_phi))

    # Get 1st PCA
    fpca_eta = sig2_eta + sig_eta_phi - lam_neg
    fpca_phi = sig2_phi + sig_eta_phi - lam_neg

    # Point PCA in direction of highest energy
    proj = fpca_eta * eta + fpca_phi * phi
    energy_up = np.sum(np.where(proj > 0, weights, 0.))
    energy_dn = np.sum(np.where(proj <= 0, weights, 0.))
    if energy_dn < energy_up:
        fpca_eta *= -1
        fpca_phi *= -1

    # Compute rotation angle
    alpha = np.pi / 2 + np.arctan(fpca_phi / fpca_eta)
    if (np.cos(alpha) * fpca_phi > np.sin(alpha) * fpca_eta):
        alpha -= np.pi

    # Flip alpha to get rotation angle
    alpha *= -1

    # Finally rotate jet
    rot_eta = eta * np.cos(alpha) - phi * np.sin(alpha)
    rot_phi = eta * np.sin(alpha) + phi * np.cos(alpha)

    return (rot_eta, rot_phi)


def flip(eta, weights):
    """ flip - This function will alter eta vector so that the majority of
    radiation described by the weights vector will sit in the positive eta
    plane.
    
    Arguments:
        eta (array) - The values of eta
        weights (array) - The weights that describe radiation pattern

    Returns
        (array) - The altered eta vector
    """

    energy_pos = np.sum(np.where(eta > 0, weights, 0.))
    energy_neg = np.sum(np.where(eta <= 0, weights, 0.))
    if energy_pos < energy_neg:
        parity = -1
    else:
        parity = 1

    return eta * parity

def normalize(vec, weights):
    """ normalize - This function normalizes the constituent data given in 
    vector by the sum of the vector given in weights.

    Arguments;
        vec (array) - values to normalize
        weights (array) - normalize by the sum of this array

    Returns:
        (array) - The normalizes values
    """
    return vec / np.sum(weights)

# -*- coding: utf-8 -*-
"""
@author: Nicolai Almskou Rasmusen &
         Victor Mølbach Nissen
"""

# %% Imports

import numpy as np
import scipy.io as scio
import functions as fun

import matplotlib.pyplot as plt

# %% functions


# %% Main

if __name__ == '__main__':
    # ---- Parameters ----
    Res = [100, 100]  # Res for [theta, tau]

    # Number of sources
    M = 5

    # Matrix dim. of given data
    N_row = 71
    N_column = 66
    freq_samples = 101

    # Subarray size
    L1 = 15  # Number of sub rows
    L2 = 15  # Number of sub columns
    L3 = 50  # Number of sub samples

    # Smoothing Array Size
    LS1 = 6
    LS2 = 6
    LS3 = 20

    # Search Space
    tau_search = [0, 5e-7]

    # plot
    plot = True

    # Chose dataset
    Data = False  # True = Experiment data, False = Simulated data

    # SNR in [dB]
    SNRdb = -10  # If set to None, noise is not added

    # ---- Initialise data ----
    # Get the index for the subarrays
    array_size = np.array([N_row, N_column, freq_samples])
    subarray_size = np.array([L1, L2, L3])
    smoothing_array_size = np.array([LS1, LS2, LS3])

    # Load datafile
    dat = scio.loadmat("data.mat")

    # Freq. domain
    if Data:
        X = dat['X']
    else:
        X = dat['X_synthetic']
        if SNRdb is not None:
            X = fun.addNoise(X, SNRdb)

    # Index data vector for antennas in subarray
    idx_array = fun.getSubarray(array_size, subarray_size,
                                offset=[0, 0, 0], spacing=2)

    # We need a L*Lf vector. Need to flatten it columnmajor (Fortran)
    X_sub = X[idx_array[0], idx_array[1]]

    # ----- Spatial Smoothing -----
    print("smooth start")
    RFB = fun.spatialSmoothing(X_sub,
                               subarray_size,
                               smoothing_array_size)

    # Need to use spatial smoothing when using MUSIC as rank is 1
    X_sub = X_sub.flatten(order='F').reshape(
        len(X_sub.flatten(order='F')), 1, order='F')
    R = X_sub @ (np.conjugate(X_sub).T)

    # ---- Run the algorithms ----
    # Do the Algorithms
    print("Algorithms")
    idx_subarray = fun.getSubarray(array_size, smoothing_array_size,
                                   offset=[0, 0, 0], spacing=2)
    print("Capon")
    Pm_Capon = fun.capon(RFB, Res, dat, idx_subarray[1],
                         idx_subarray[0], tau_search)

    print("Bartlett")
    Pm_Bartlett = fun.bartlett(R, Res, dat, idx_array[1],
                               idx_array[0], tau_search)

    print("MUSIC")
    Pm_MUSIC = fun.MUSIC(RFB, Res, dat, idx_subarray[1],
                         idx_subarray[0], tau_search, M=M)

    # %% Plot
    if plot:
        AoA = (dat['smc_param'][0][0][1])*180/np.pi
        AoA[AoA < 0] = AoA[AoA < 0] + 360

        TDoA = (dat['smc_param'][0][0][2])*(1/3e8) + np.abs(dat['tau'][0])

        plt.figure()
        # plt.title(f"Capon - res: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='o', facecolors='none')
        pm_max = np.max(10*np.log10(Pm_Capon))
        plt.imshow(10*np.log10(Pm_Capon), vmin=pm_max-40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0], tau_search[1]],
                   aspect="auto", origin="lower")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")
        plt.savefig("CAPON_test.pdf")

        plt.figure()
        # plt.title(f"Bartlett - res: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='o', facecolors='none')
        pm_max = np.max(10*np.log10(Pm_Bartlett))
        plt.imshow(10*np.log10(Pm_Bartlett), vmin=pm_max-40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0], tau_search[1]],
                   aspect="auto", origin="lower")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")
        plt.savefig("BARTLETT_test.pdf")

        plt.figure()
        # plt.title(f"MUSIC - res: {Res} - SNRdb: {SNRdb}")
        plt.scatter(AoA, TDoA, color='r', marker='o', facecolors='none')
        pm_max = np.max(10*np.log10(Pm_MUSIC))
        plt.imshow(10*np.log10(Pm_MUSIC), vmin=pm_max-40, vmax=pm_max,
                   extent=[0, 360,
                           tau_search[0], tau_search[1]],
                   aspect="auto", origin="lower")
        plt.colorbar()
        plt.ylabel("Tau [s]")
        plt.xlabel("Theta [degrees]")
        plt.savefig("MUSIC_test.pdf")

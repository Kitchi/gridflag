"""
Module to hold all the histogram related functions.
"""

import numba as nb
import numpy as np
from xradio.vis.read_processing_set import read_processing_set
from xradio.vis.load_processing_set import load_processing_set


def create_uv_histogram(input_params):
    """
    Read in the input PS and calculate a histogram per pixel in the UV plane.
    """

    ps_name = input_params['input_data_name']
    ps = read_processing_set(ps_name = ps_name)
    main_xds = ps.get(0)
    # in Hz
    ref_freq = main_xds.frequency.attrs['reference_frequency']['data']

    uvrange = sorted(input_params['uvrange'])
    uvcell = input_params['uvcell']
    nhistbin = input_params['nhistbin']
    npixu, npixv = input_params['npixels']

    for msv4name, slice_description in input_params['data_sel'].items():
        ps = load_processing_set(ps_name = ps_name, sel_parms={msv4name:slice_description})

        ms_xds = ps.get(0)
        print(ms_xds.VISIBILITY.shape)
        print(ms_xds.UVW.shape)
        # Drop rows that are outside the selected UV range
        ms_xds = ms_xds.where((ms_xds.UVW[...,0]**2 + ms_xds.UVW[...,1]**2 < uvrange[1]*uvrange[1]), drop=True)
        print(ms_xds.VISIBILITY.shape)
        print('-----------------')
        uvw = ms_xds.UVW.data
        vis = ms_xds.VISIBILITY.data
        flag = ms_xds.FLAG.data.astype(bool)
        freq = ms_xds.frequency.data

        # Fill NaNs with ones, so when it gets flipped to multiply the
        # visibilities it flags the corresponding visibilities.
        flag = np.nan_to_num(flag, nan=1)
        vis = np.nan_to_num(vis)
        # Flag visibilities
        vis = vis*~flag

        uvw = np.nan_to_num(uvw)
        uv_scaled = scale_uv_freq(uvw, freq, ref_freq)

        # Flatten out UV array
        # Manually verified that this works for a handful of random indices
        ms_uv_hist = uv_histogram(uv_scaled.reshape([-1,2]), vis.reshape([-1,2]), uvrange, uvcell, npixu, npixv, nhistbin)

        #print(uvw.shape, vis.shape, flag.shape, freq.shape, uv_scaled.shape)


@nb.jit(nopython=True, nogil=True, cache=True)
def hermitian_conjugate(uv, vis):
    """
    Given the input UV coordinates and visibilities, calculate the hermitian conjugate
    of the visibilities.
    """

    nvis = vis.shape[0]
    for nn in range(nvis):
        if uv[nn, 1] < 0:
            uv[nn, 1] *= -1
            vis[nn] = vis[nn].conj()

    return uv, vis



#@nb.jit(nopython=True, nogil=True, cache=True)
def uv_histogram(uv, vis, uvrange, uvcell, npixu, npixv, nhistbin=100):
    """
    Generate a histogram per UV pixel, given the input UV coordinates & visibilities.
    """


    uv, vis = hermitian_conjugate(uv, vis)

    uvmax = max(uvrange)

    # Calculate the UV bin for each point
    # Note - these are floats at this point, need to cast to int to index
    ubin = (uv[:, 0] + uvmax)//uvcell
    vbin = uv[:, 1]//uvcell

    # Create a histogram per UV pixel - some might be entirely zeros, with no data.
    vis_hist = np.zeros((npixu, npixv, nhistbin), dtype=int)
    # Store the bin values for each pixel
    vis_hist_bins = np.zeros((npixu, npixv, nhistbin), dtype=float)

    for ridx, row in enumerate(vis):
        # Amplitude of Stokes I for now
        # TODO : Write support for other quantities
        stokesI = np.abs((row[...,0] + row[...,-1])/2.)
        hist, bins = np.histogram(stokesI, bins=nhistbin)
        bins = (bins[1:] + bins[:-1])/2.

        uu = int(ubin[ridx])
        vv = int(vbin[ridx])

        if np.count_nonzero(vis_hist_bins[uu, vv]) == 0:
            vis_hist_bins[uu, vv] = bins.copy()

        vis_hist_bins[uu,vv], vis_hist[uu, vv] = hist_merge(bins, hist, vis_hist_bins[uu,vv], vis_hist[uu,vv], nbin=nhistbin)


@nb.jit(nopython=True, nogil=True, cache=True)
def hist_merge(bin1, hist1, bin2, hist2, nbin=100):
    """
    Merge two input histograms with differing binning.
    """

    min_bin = np.asarray([bin1.min(), bin2.min()]).min()
    max_bin = np.asarray([bin1.max(), bin2.max()]).max()

    bins = np.linspace(min_bin, max_bin, nbin)
    merge_hist = np.zeros(nbin)
    delta = bins[2] - bins[1]

    for bidx, bb in enumerate(bin1):
        idx = int(np.floor((bb - min_bin) / delta))

        if idx < 0:
            idx = 0
        elif idx >= nbin:
            idx = nbin - 1

        merge_hist[idx] += hist1[bidx]

    for bidx, bb in enumerate(bin2):
        idx = int(np.floor((bb - min_bin) / delta))

        if idx < 0:
            idx = 0
        elif idx >= nbin:
            idx = nbin - 1

        merge_hist[idx] += hist2[bidx]

    return bins, merge_hist


@nb.jit(nopython=True, nogil=True, cache=True)
def centre_bins(bins):
    bins = (bins[:-1] + bins[1:])/2.
    w = 0.9 * (bins[1] - bins[0])

    return bins, w


@nb.jit(nopython=True, nogil=True, cache=True)
def scale_uv_freq(uvw, frequency, ref_freq):
    """
    Given the input UVW and frequency coordinates, scale the UVW coordinates
    correctly per frequency.

    This only returns the scaled U and V values - it drops W.
    """

    shape = uvw.shape
    uv_scaled = np.zeros((shape[0], shape[1], frequency.size, 2))

    for ffidx, ff in enumerate(frequency):
        delta_nu = (ff - ref_freq)/ref_freq
        uv_scaled[:,:,ffidx,0] = uvw[:,:,0] * (1 + delta_nu/ff)
        uv_scaled[:,:,ffidx,1] = uvw[:,:,1] * (1 + delta_nu/ff)

    return uv_scaled


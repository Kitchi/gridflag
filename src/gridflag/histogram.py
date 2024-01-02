"""
Module to hold all the histogram related functions.
"""

import numba as nb
import numpy as np
from xradio.vis.read_processing_set import read_processing_set
from xradio.vis.load_processing_set import load_processing_set

import dask

import logging

logging.addLevelName(logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)-20s %(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler("plumber.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()


@nb.jit(nopython=True, nogil=True, cache=True)
def _merge_accum_hist(ms_uv_hist_bins, ms_uv_hist, accum_uv_hist_bins, accum_uv_hist, npixu, npixv, nhistbin):
    for uidx in range(npixu):
        for vidx in range(npixv):
            if np.count_nonzero(accum_uv_hist_bins[uidx, vidx]) == 0:
                accum_uv_hist_bins[uidx, vidx] = ms_uv_hist_bins[uidx, vidx].copy()

            accum_uv_hist_bins[uidx, vidx], accum_uv_hist[uidx, vidx] = hist_merge(ms_uv_hist_bins[uidx, vidx], ms_uv_hist[uidx, vidx],
                                                                                   accum_uv_hist_bins[uidx, vidx], accum_uv_hist[uidx, vidx], nbin=nhistbin)

    return accum_uv_hist_bins, accum_uv_hist


@dask.delayed
def compute_uv_histogram(ms_xds, input_params):
    """
    Read in the input XDS and calculate a histogram per pixel in the UV plane.
    """

    # in Hz
    ref_freq = float(ms_xds.frequency.attrs['reference_frequency']['data'])

    uvrange = np.asarray(sorted(input_params['uvrange']))
    uvcell = input_params['uvcell']
    nhistbin = input_params['nhistbin']
    npixu, npixv = input_params['npixels']

    # Create a histogram per UV pixel - some might be entirely zeros, with no data.
    accum_uv_hist = np.zeros((npixu, npixv, nhistbin), dtype=int)
    # Store the histogram bin values for each pixel
    accum_uv_hist_bins = np.zeros((npixu, npixv, nhistbin), dtype=float)

    min_baseline = ms_xds.baseline_id.min().data
    max_baseline = ms_xds.baseline_id.max().data

    print(ms_xds.VISIBILITY.data.shape)
    # Drop rows that are outside the selected UV range
    ms_xds = ms_xds.where(((ms_xds.UVW[...,0]**2 + ms_xds.UVW[...,1]**2 > uvrange[0]*uvrange[0]) & (ms_xds.UVW[...,0]**2 + ms_xds.UVW[...,1]**2 < uvrange[1]*uvrange[1])), drop=True)
    print(ms_xds)
    print(ms_xds.VISIBILITY.data.shape)

    uvw = ms_xds.UVW.data
    print("nonzero counts ", np.count_nonzero(uvw))
    vis = ms_xds.VISIBILITY.data
    print("nonzero counts ", np.count_nonzero(vis))
    flag = ms_xds.FLAG.data.astype(bool)
    print("nonzero counts ", np.count_nonzero(flag))
    freq = ms_xds.frequency.data
    print("nonzero counts ", np.count_nonzero(freq))

    # Flip flags and replace NaNs with zeros, so they flag the corresponding visibilities
    flag = np.nan_to_num(~flag).astype(bool)
    vis = np.nan_to_num(vis)
    # Flag visibilities
    #vis = np.asarray(vis*~flag)

    print("nonzero counts ", np.count_nonzero(vis))

    uvw = np.nan_to_num(uvw)
    uv_scaled = scale_uv_freq(np.asarray(uvw), np.asarray(freq), ref_freq)

    # Create a histogram per UV pixel - some might be entirely zeros, with no data.
    vis_hist = np.zeros((npixu, npixv, nhistbin), dtype=int)
    # Store the bin values for each pixel
    vis_hist_bins = np.zeros((npixu, npixv, nhistbin), dtype=float)

    # Manually verified that the reshape works for a handful of random indices
    ms_uv_hist_bins, ms_uv_hist = uv_histogram(uv_scaled.reshape([-1,2]), vis.reshape([-1,2]), uvrange, uvcell, npixu, npixv, vis_hist, vis_hist_bins, nhistbin)
    print("internal nonzero counts mshist ", np.count_nonzero(ms_uv_hist))
    print("internal nonzero counts msbins ", np.count_nonzero(ms_uv_hist_bins))

    if np.count_nonzero(accum_uv_hist_bins) == 0:
        accum_uv_hist_bins = ms_uv_hist_bins.copy()

    print("Merging histograms")

    #accum_uv_hist_bins, accum_uv_hist = hist_merge(ms_uv_hist_bins, ms_uv_hist, accum_uv_hist_bins, accum_uv_hist, nbin=nhistbin)
    accum_uv_hist_bins, accum_uv_hist = _merge_accum_hist(ms_uv_hist_bins, ms_uv_hist, accum_uv_hist_bins, accum_uv_hist, npixu, npixv, nhistbin)

    print("internal nonzero counts hist ", np.count_nonzero(accum_uv_hist))
    print("internal nonzero counts bins ", np.count_nonzero(accum_uv_hist_bins))

    del ms_uv_hist_bins, ms_uv_hist

    return accum_uv_hist_bins, accum_uv_hist



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



@nb.jit(nopython=True, nogil=True, cache=True)
def uv_histogram(uv, vis, uvrange, uvcell, npixu, npixv, vis_hist, vis_hist_bins, nhistbin=100):
    """
    Generate a histogram per UV pixel, given the input UV coordinates & visibilities.
    """


    uv, vis = hermitian_conjugate(np.asarray(uv), np.asarray(vis))

    uvmax = max(uvrange)

    # Calculate the UV bin for each point
    # Note - these are floats at this point, need to cast to int to index
    ubin = (uv[:, 0] + uvmax)//uvcell
    vbin = uv[:, 1]//uvcell

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

    print("very internal nonzero counts hist ", np.count_nonzero(vis_hist))
    print("very internal nonzero counts bins ", np.count_nonzero(vis_hist_bins))

    #import uuid
    #tmpname = str(uuid.uuid4())
    #np.savez(tmpname, vis_hist_bins=vis_hist_bins, vis_hist=vis_hist)

    return vis_hist_bins, vis_hist


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

    if delta == 0:
        return bins, merge_hist

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



@nb.jit(nopython=True, nogil=True, cache=True)
def histogram_median(hist, bins):
    """
    Calculate the median of a given histogram.
    Note : The bin values here are assumed to be the left edge of the bin.

    Inputs:
    hist        : Histogram counts (1D array)
    bins        : Bin values (1D array)

    Returns:
    median      : Median of the histogram, float
    """

    # Formula stolen from here : https://math.stackexchange.com/questions/879052/how-to-find-mean-and-median-from-histogram

    hist_csum = np.cumsum(hist)
    hsum = hist.sum()//2

    median_bin_idx = np.argmin(np.abs(hist_csum - hsum))
    halfcsum = hist_csum[median_bin_idx - 1]

    lbin = bins[median_bin_idx]
    fbin = hist[median_bin_idx]

    if fbin == 0:
        return 0

    w = bins[median_bin_idx + 1] - bins[median_bin_idx]

    median = lbin + ((hsum - halfcsum)/fbin)*w

    return median


@nb.jit(nopython=True, nogil=True, cache=True)
def histogram_std(hist, bins):
    """
    Calculate the standard deviation of a histogram.
    Note : The bin values here are assumed to be the left edge of the bin.

    Inputs:
    hist        : Histogram counts (1D array)
    bins        : Bin values (1D array)

    Returns:
    std         : Standard deviation of the histogram, float
    """

    median = histogram_median(hist, bins)
    hsum = hist.sum()

    if hsum == 0:
        return 0

    # Calculate the standard deviation
    std = np.sqrt(np.sum(hist*(bins - median)**2)/hsum)

    return std



@nb.jit(nopython=True, nogil=True, cache=True)
def histogram_med_std(hist, bins):
    """
    Calculate the median and standard deviation of a histogram.
    """

    median = histogram_median(hist, bins)
    std = histogram_std(hist, bins)

    return median, std



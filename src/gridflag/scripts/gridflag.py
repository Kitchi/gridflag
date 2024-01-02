#!/usr/bin/env python3

import click
import logging

import numpy as np
import numba as nb

from xradio.vis.read_processing_set import read_processing_set
from xradio.vis.load_processing_set import load_processing_set

import xarray as xr

from gridflag.histogram import compute_uv_histogram, _merge_accum_hist

import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
dask.config.set(scheduler='synchronous')

#Add colours for warnings and errors
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



def create_chunks(baseline_id, frequencies, baseline_chunk_size, freq_chunk_size):
    """
    Given the input baseline_ids and frequencies, return a list of the start and end slices
    to parallelize over.

    Inputs:
    baseline_id             : Baseline IDs, np.array
    frequencies             : Frequencies, np.array
    baseline_chunk_size     : Number of baseline IDs to process per chunk, int
    freq_chunk_size         : Number of frequencies to process per chunk, int

    Returns:
    baseline_chunks         : List of start and end slices for baseline IDs
    freq_chunks             : List of start and end slices for frequencies
    """

    baseline_chunks = []
    freq_chunks = []

    nbaseline = len(baseline_id)
    nfreq = len(frequencies)

    for bb in range(0, len(baseline_id), baseline_chunk_size):
        if bb+baseline_chunk_size > nbaseline:
            baseline_chunks.append(slice(baseline_id[bb], None))
        else:
            baseline_chunks.append(slice(baseline_id[bb],baseline_id[bb+baseline_chunk_size]))

    for ff in range(0, len(frequencies), freq_chunk_size):
        if ff+freq_chunk_size > nfreq:
            freq_chunks.append(slice(frequencies[ff], None))
        else:
            freq_chunks.append(slice(frequencies[ff],frequencies[ff+freq_chunk_size]))

    return baseline_chunks, freq_chunks



ctx = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=ctx)
@click.argument('zarr', type=click.Path(exists=True))
@click.option('--uvrange', nargs=2, type=int, default=[None, None], show_default=True, help='UV range within which to flag')
@click.option('--uvcell', default=None, type=int, show_default=True, help='UV cell size to use for flagging.')
@click.option('--nhistbin', default=100, type=int, show_default=True, help='Number of bins in the visibility histogram per bin')
@click.option('--nsigma', default=5, type=int, show_default=True, help='Sigma threshold for flagging')
def main(zarr, uvrange, uvcell, nhistbin, nsigma):
    """
    Given the input visibility data set (in Zarr format), flags the data using
    the GRIDflag algorithm.

    Please always specify --uvrange and --uvcell. Future versions will attempt
    to auto-discover these parameters but it is currently not implemented.
    """

    baseline_chunk_size = 128
    freq_chunk_size = 16

    #client = Client()
    #client = Client(LocalCluster(n_workers=4, threads_per_worker=1))

    # Lazy load the processing set
    ps = read_processing_set(zarr)

    npixu = 2*uvrange[1]//uvcell
    npixv = uvrange[1]//uvcell

    input_params = {}
    input_params['uvrange'] = uvrange
    input_params['uvcell'] = uvcell
    input_params['nhistbin'] = nhistbin
    input_params['npixels'] = [npixu, npixv]
    input_params['input_data'] = zarr

    # TO DO : Investigate using sparse matrices to help memory usage here...
    output = []
    for idx in range(len(ps)):
        # Chunk up each MSv4 in the processing set independently
        ds = ps.get(idx)
        baseline_id = ds.coords['baseline_id']
        frequencies = ds.coords['frequency']

        baseline_chunks, freq_chunks = create_chunks(baseline_id, frequencies, baseline_chunk_size, freq_chunk_size)

        for bb in range(len(baseline_chunks)):
            for ff in range(len(freq_chunks)):
                print(bb, ff)
                loc_dict = {'baseline_id':baseline_chunks[bb], 'frequency':freq_chunks[ff]}
                this_ds = ds.loc[loc_dict]
                print(this_ds)
                input()

                this_output = compute_uv_histogram(this_ds, input_params)
                output.append(this_output)

    result = dask.compute(output)

    # Note :
    # result is a list of lists, the inner "list" is itself a list of lists...
    # This contains the histogram and bins for each baseline and frequency chunk above
    # So len(result[0]) == number of baseline chunks * number of frequency chunks
    # We now need to iterate through result[0] and sum the histograms and bins together

    nchunk = len(output)
    # Create a histogram per UV pixel - some might be entirely zeros, with no data.
    accum_uv_hist = np.zeros((npixu, npixv, nhistbin), dtype=int)
    # Store the histogram bin values for each pixel
    accum_uv_hist_bins = np.zeros((npixu, npixv, nhistbin), dtype=float)

    for nn in range(nchunk):
        print('nonzero counts bins ', np.count_nonzero(result[0][nn][0]))
        print('nonzero counts hist ', np.count_nonzero(result[0][nn][1]))
        accum_uv_hist_bins, accum_uv_hist = _merge_accum_hist(result[0][nn][0], result[0][nn][1], accum_uv_hist_bins, accum_uv_hist, npixu, npixv, nhistbin)

    print("nonzero counts ", np.count_nonzero(accum_uv_hist))

    import uuid
    tmpname = str(uuid.uuid4())
    np.savez(tmpname, vis_hist_bins=accum_uv_hist_bins, vis_hist=accum_uv_hist)

    print("FINISHED COMPUTE")
    #client.close()



if __name__ == '__main__':
    main()

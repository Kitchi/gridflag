#!/usr/bin/env python3

import click
import logging

import numpy as np
import numba as nb

from xradio.vis.read_processing_set import read_processing_set
from xradio.vis.load_processing_set import load_processing_set

from astroviper._concurrency._graph_tools import _make_parallel_coord, _map
from astroviper.client import local_client

from gridflag.histogram import *

import dask
dask.config.set(scheduler='synchronous')

# Add colours for warnings and errors
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

    # Lazy load the processing set
    ps = read_processing_set(zarr)

    parallel_coords = {}
    n_chunks = 8
    parallel_coords['baseline_id'] = _make_parallel_coord(coord=ps.get(0).baseline_id, n_chunks=n_chunks)

    n_chunks = 6
    parallel_coords['frequency'] = _make_parallel_coord(coord=ps.get(0).frequency, n_chunks=n_chunks)

    npixu = 2*uvrange[1]//uvcell
    npixv = uvrange[1]//uvcell

    input_params = {}
    input_params['uvrange'] = uvrange
    input_params['uvcell'] = uvcell
    input_params['nhistbin'] = nhistbin
    input_params['npixels'] = [npixu, npixv]

    sel_params = {}
    sel_params['fields'] = ['1453+330']
    sel_params['intents'] = None


    graph = _map(input_data_name = zarr,
                 input_data_type='processing_set',
                 parallel_coords = parallel_coords,
                 input_parms = input_params,
                 ps_sel_parms = sel_params,
                 func_chunk = create_uv_histogram,
                 client = local_client)

    dask.compute(graph)

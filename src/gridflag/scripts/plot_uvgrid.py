#!/usr/bin/env python3

import click

import numpy as np
import numba as nb
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, CustomJS

from gridflag.histogram import histogram_med_std

import bokeh.palettes as bp


@nb.jit(nopython=True, nogil=True, cache=True)
def calc_median_std_grid(uvhist, uvhistbins):
    """
    Given the 2D histogram, calculate median and standard deviation per uv-cell

    Inputs:
    uvhist      : 2D histogram, shape (npixu, npixv, nhistbin)
    uvhistbins  : Bin values, shape (npixu, npixv, nhistbin)

    Returns:
    median      : Median of the histogram, shape (npixu, npixv)
    std         : Standard deviation of the histogram, shape (npixu, npixv)
    """

    npixu = uvhist.shape[0]
    npixv = uvhist.shape[1]

    median = np.zeros((npixu, npixv))
    std = np.zeros((npixu, npixv))

    for uu in range(npixu):
        for vv in range(npixv):
            median[uu, vv], std[uu, vv] = histogram_med_std(uvhist[uu, vv], uvhistbins[uu, vv])

    return median, std



ctx = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=ctx)
@click.argument('griddata', type=click.Path(exists=True))
def main(griddata) :
    """
    Given an input NPZ file containing 2x 3D UV grid data - (U, V, histogram)
    (U, V, hist_bins) this script plots the median and standard deviation
    gridded data.

    The keys in the NPZ must be :

    'vis_hist' : Contains the histogram counts, shape (npixu, npixv, nhistbin)
    'vis_hist_bins' : Contains the bin values, shape (npixu, npixv, nhistbin)
    """

    uvdict = np.load(griddata)

    uvhist = uvdict['vis_hist']
    uvhistbins = uvdict['vis_hist_bins']

    # Calculate the median and standard deviation
    hist_median, hist_std = calc_median_std_grid(uvhist, uvhistbins)


    # Plot the median and standard deviation
    output_file("grid_median_std.html")

    # Create a figure
    p1 = figure(title="Median", tools="hover", toolbar_location=None)
    p1.image(image=[hist_median], x=0, y=0, dw=uvhist.shape[0], dh=uvhist.shape[1], palette="Spectral11")

    p2 = figure(title="Standard Deviation", tools="hover", toolbar_location=None)
    p2.image(image=[hist_std], x=0, y=0, dw=uvhist.shape[0], dh=uvhist.shape[1], palette="Spectral11")

    p = gridplot([[p1, p2]])

    show(p)

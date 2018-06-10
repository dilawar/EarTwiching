#!/usr/bin/env python3

import os
import sys
import pandas as pd
import method_mean_std as _method
import pickle
import numpy as np
import multiprocessing
import glob
import helper

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use( 'seaborn-talk' )
mpl.rcParams['axes.linewidth'] = 0.2
mpl.rcParams['text.usetex'] = False
trial_data_ = [ ]

def generate_pickle( tif_file ):
    _method.run( tif_file )

def generate_all_pickle( trialdir, worker = 3 ):
    global trial_data_

    pks = glob.glob( "%s/trial*.pkl" % trialdir )
    tiffs = glob.glob( "%s/trial*.tif" % trialdir)
    tiffs += glob.glob( "%s/trial*.tif?" % trialdir)

    tiffs = sorted( tiffs )
    pks = sorted( pks )
    print( "[INFO ] Total %s files found" % len(tiffs) )

    # remove files for which pkl file is generated.
    newtiffs = []
    for tfile in tiffs:
        b = os.path.realpath(tfile)
        a =  b + '.pkl'
        if os.path.isfile( a ):
            print( "[INFO ] Pickle is already generated. Ignoring %s" % tfile )
        else:
            newtiffs.append(tfile)

    if worker > 1:
        pool = multiprocessing.Pool( worker )
        pool.map( generate_pickle, newtiffs )
    else:
        for f in newtiffs:
            generate_pickle(f)
        print( "[INFO ] Pickle generation is over." )

def read_pickle( datadir ):
    pks = sorted( glob.glob( "%s/*.pkl" % datadir) )
    data = []
    for p in pks:
        with open(p, 'rb') as f:
            d = pickle.load( f )
            data.append( d )
    return data

def plot_summary_data( data ):
    res = []
    tmin, tmax, nMax = None, None, 0
    for mask, summaryImg, lines in data:
        d = helper.lines_to_dataframe( lines )
        x, y = d['t1'].values, d['sig2'].values 

        y =  np.abs(y - np.mean(y[:20]))
        y = y / y.max()

        tmax = helper._max(tmax, x.max())
        tmin = helper._min(tmin, x.min())
        nMax = helper._max(nMax, len(y))
        res.append( (x,y) )

    img = []
    for x, y in res:
        a = np.array(x, dtype=float)
        a0 = np.linspace( min(a), max(a), nMax)
        y0 = np.interp(a0, a, y)
        img.append(y0)

    plt.imshow(img, interpolation = 'none', aspect = 'auto')
    plt.colorbar()
    plt.savefig( 'summary.png' )

def main( ):
    datadir = sys.argv[1]
    print( '[INFO] Processing %s' % datadir )
    generate_all_pickle( datadir, 3 )
    data = read_pickle( datadir )
    plot_summary_data( data )

if __name__ == '__main__':
    main()
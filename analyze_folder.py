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
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    mpl.style.use( 'seaborn-talk' )
except Exception as e:
    pass
mpl.rcParams['axes.linewidth'] = 0.2
mpl.rcParams['text.usetex'] = False
trial_data_ = [ ]

def generate_pickle( tif_file):
    try:
        _method.run( tif_file)
    except Exception as e:
        print( '[WARN] Failed to process %s. Reason %s' % (tif_file,e))

def generate_all_pickle( trialdir, worker = 3 ):
    global trial_data_

    pks = glob.glob( "%s/_results/trial*.pkl" % trialdir )
    tiffs = glob.glob( "%s/trial*.tif" % trialdir)
    tiffs += glob.glob( "%s/trial*.tif?" % trialdir)

    tiffs = sorted( tiffs )
    pks = sorted( pks )
    print( "[INFO ] Total %s files found" % len(tiffs) )

    # remove files for which pkl file is generated.
    newtiffs = []
    pksname = [os.path.basename(x) for x in pks]
    for tfile in tiffs:
        b = os.path.basename(os.path.realpath(tfile))
        if ('%s.pkl' % b) in pksname:
            print( "[INFO ] Pickle is already generated. Ignoring %s" % tfile )
        else:
            newtiffs.append(tfile)

    print( '[INFO] Total tiff files to process %s' % len(newtiffs))
    if worker > 1:
        pool = multiprocessing.Pool( worker )
        pool.map( generate_pickle, newtiffs )
    else:
        for f in newtiffs:
            generate_pickle(f)
        print( "[INFO ] Pickle generation is over." )

def read_pickle( datadir ):
    pks = sorted( glob.glob( "%s/_results/*.pkl" % datadir) )
    data = []
    for p in pks:
        with open(p, 'rb') as f:
            d = pickle.load( f )
            data.append( d )
    return data

def timestamp_to_str(ts):
    return [ datetime.datetime.fromtimestamp(ns/1e9).strftime('%M:%S.%f') 
            for ns in ts ]

def plot_summary_data( data, outfile ):
    res = []
    tmin, tmax, nMax = None, None, 0
    print( len(data) )

    csStartTimes = [ ]
    for mask, summaryImg, lines in data:
        d = helper.lines_to_dataframe( lines )
        x, y = d['t1'].values, d['sig2'].values 
        isProbe = 'PUFF' not in list(d['status'])
        csdata = d[ d['status'] == 'CS+' ]
        if len(csdata) < 1:
            print( '[WARNING] Empy signal from tiff file. Ignoring..' )
            continue
        csStartTime = np.float( csdata['t1'].values[0] )
        csStartTimes.append( csStartTime )

        y -= y.min()
        y /= y.max()

        # Subtract baseline.
        y -= np.mean(y[:30])

        tmax = helper._max(tmax, x.max())
        tmin = helper._min(tmin, x.min())
        nMax = helper._max(nMax, len(y))
        res.append( (x,y, isProbe) )

    img, imgProbe, imgX = [], [], []
    X1, X2 = [], []
    print( ' -> NMAX %d' % nMax )
    for i, (x, y, isProbe) in enumerate(res):
        a = np.array(x, dtype=float) - csStartTimes[i]  # in nano-seconds.
        a = a / 1e6                                  # in milli-seconds.
        a0 = np.linspace( -400, 800, nMax)
        y0 = np.interp(a0, a, y)
        imgX.append( a0 )
        if isProbe:
            imgProbe.append(y0)
        else:
            img.append(y0)

    imgx = np.mean(imgX, axis=0)
    plt.subplot(221)
    plt.imshow(img, interpolation = 'none', aspect = 'auto')
    plt.colorbar()
    plt.xticks( np.arange(0, len(imgx), 20)
            , [ '%d' % x for x in imgx[::20] ]
            , rotation=90 )

    plt.subplot(222)
    y, yerr = np.mean(img, axis=0), np.std(img, axis=0)
    plt.plot( a0, y )
    #plt.xticks( a0[::10], timestamp_to_str(a0)[::10] )
    plt.fill_between( a0, y+yerr, y-yerr, alpha = 0.2 )


    plt.subplot(223)
    plt.imshow(imgProbe, interpolation = 'none', aspect = 'auto' )
    plt.colorbar( )
    plt.xticks( np.arange(0, len(imgx), 20)
            , [ '%d' % x for x in imgx[::20] ]
            , rotation=90 )
    plt.title( 'PROBE Trials' )

    plt.subplot(224)
    y, yerr = np.mean(imgProbe, axis=0), np.std(imgProbe, axis=0)
    plt.plot( a0, y )
    plt.fill_between( a0, y+yerr, y-yerr, alpha = 0.2 )



    plt.tight_layout( )
    plt.savefig( outfile )
    print( "[INFO ] Saved to %s" % outfile )

def main( ):
    datadir = sys.argv[1]
    nworks = 4
    if len(sys.argv) > 2:
        nworks = int(sys.argv[2])
    print( '[INFO] Processing %s' % datadir )
    generate_all_pickle( datadir, nworks )
    data = read_pickle( datadir )
    outfile = os.path.join( datadir, 'summary.png' )
    plot_summary_data( data, outfile )

if __name__ == '__main__':
    main()

#!/usr/bin/env python

import os
import sys
import tifffile
import numpy as np
import itertools
import pickle
import dateutil.parser 
import helper

cap_ = None

resdir_name_ = '_results'
datadir_     = None
infile_      = None
frames_      = []

def fetch_a_good_frame( drop = 0 ):
    global cap_
    return next(cap_).asarray()

def read_all_frames( infile ):
    global cap_
    print( "[INFO ] Reading frames from %s" % infile )
    cap_ = (x for x in tifffile.TiffFile( infile ).pages )
    assert cap_
    frames = []
    while True:
        f = None
        try:
            f = fetch_a_good_frame( )
        except Exception as e:
            pass

        if f is None:
            break
        frames.append( f )
    print( "[INFO ] Fetched %d frames"  % len(frames) )
    return np.array( frames )


def preprocess_all_frames(frames):
    global infile_, datadir_

    assert os.path.isfile( infile_ ), "%s not fonund" % infile_
    infname = os.path.basename(infile_)

    meanFrame = np.mean(frames, axis=0)
    threshold = np.zeros_like(meanFrame)
    stdFrame = np.std(frames, axis = 0)
    r, c = meanFrame.shape
    thres = 30 
    totalPixels = np.product( meanFrame.shape )
    o = 100
    for i, j in itertools.product(range(o,r-o), range(o,c-o)):
        n = i*c + j
        if n % 10000 == 0:
            print( '[INFO] Pixel %d/%d are done' % (n, totalPixels))

        vec = frames[:,i,j]
        u = np.mean(vec)
        v = np.std(vec)

        # This is critical. The pixel in ear are slightly darker and they become
        # whiter. That means we are only looking for pixels where changes are
        # towards high value. That is the mean is smaller than the max.
        # rest to zeros.
        if u < vec.max() and u+1.2*v > vec.max():
            threshold[i,j] = 255

    mask = np.where( threshold == 255 )
    lines = process( frames, mask )
    return mask, [meanFrame, stdFrame, threshold], lines

def process( frames, mask ):
    lines = []
    for f in frames:
        signal = np.mean(f[mask])
        fs = ''.join( [ chr(int(x)) for x in f[0,:] ]).rstrip()
        fs += ',%g' % signal
        lines.append( fs)
    return '\n'.join(lines)

def plot_trial(summary, lines, outfile ):
    import io
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.dates as md

    mpl.rcParams['text.usetex'] = False

    plt.figure( figsize=(12,6) )
    gridSize = (5, 12)
    ax1 = plt.subplot2grid( gridSize, (0,0), colspan = 4, rowspan=2 )
    ax2 = plt.subplot2grid( gridSize, (0,4), colspan = 4, rowspan=2 )
    ax3 = plt.subplot2grid( gridSize, (0,8), colspan = 4, rowspan=2 )
    ax4 = plt.subplot2grid( gridSize, (2,0), colspan = 12, rowspan=2 )

    ax1.imshow( summary[0] )
    ax1.set_title( 'Mean' )
    ax2.imshow( summary[1] )
    ax2.set_title( 'Std' )
    ax3.imshow( summary[2] )
    ax3.set_title( 'Pixels of interest' )

    d = helper.lines_to_dataframe(lines)
    x, y = d['t1'], d['sig2']
    #  x = [ dateutil.parser.parse(a).microsecond for a in x ]
    ax4.plot( x, y, lw = 2)
    #  ax4.xaxis.set_major_formatter( md.DateFormatter('%M %S') )
    #  ax4.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax4.set_xlabel( 'Time' )

    ymin = np.min(y)
    for s in [ 'CS+', 'PUFF']:
        t0, t1 = helper.get_time_slice(d, s)
        if t1 > t0:
            ax4.plot( [t0, t1], [ ymin, ymin], lw = 3)
    ax4.set_ylabel( 'Signal' )

    # Sample frames on ax5.
    scale = int(len(frames_)/12) - 1
    for i in range(12):
        ax = plt.subplot2grid( gridSize, (4,i), colspan = 1 )
        ax.set_axis_off()
        ax.imshow( frames_[scale*i], aspect='auto' )

    plt.suptitle( 'Total frames %d' % len(x) )
    plt.tight_layout( h_pad=0, w_pad=0)
    plt.savefig( outfile )

def run( infile, ignore_pickle = False ):
    global datadir_, infile_
    global frames_
    infile_ = infile

    print( '[INFO] Ignore pickler? %s' % ignore_pickle )
    datadir_ = os.path.join( os.path.dirname( infile_ ), resdir_name_ )
    infilename = os.path.basename( infile_ )

    if not os.path.exists( datadir_ ):
        os.makedirs( datadir_ )

    picklefile = os.path.join(datadir_, '%s.pkl' % infilename)
    frames_ = read_all_frames( infile_ )
    if not os.path.exists( picklefile ) or ignore_pickle:
        res = preprocess_all_frames( frames_ )
        with open( picklefile, 'wb') as f:
            pickle.dump( res, f )
            print( "[INFO ] Wrote to picklefile %s" % picklefile )

    # Pickfile is found. load it.
    with open( picklefile, 'rb') as f:
        threshold, summaryImg, lines = pickle.load( f )
        plot_trial(summaryImg, lines, os.path.join( datadir_, '%s.signal.png' % infilename) )

    return dict(threshold=threshold, summary_img=summaryImg, lines=lines)
    
def main():
    global infile_
    infile_ = sys.argv[1]
    ignore_pickle = False
    if '--force' in sys.argv:
        ignore_pickle = True
    run( infile_, ignore_pickle = ignore_pickle)

if __name__ == '__main__':
    main()

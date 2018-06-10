#!/usr/bin/env python

import os
import sys
import tifffile
import numpy as np
import itertools
import pickle
import dateutil.parser 
import pandas as pd

cap_ = None

resdir_name_ = '_results'
datadir_ = None
infile_ = None

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

def preprocess_all_frames(frames, outfile = None):
    global infile_, datadir_
    infname = os.path.basename( infile_ )

    meanFrame = np.mean( frames, axis = 0 )
    threshold = np.zeros_like(meanFrame)
    stdFrame = np.std( frames, axis = 0 )
    newframes = frames[:]

    #  newframes[np.where(frames < (meanFrame+stdFrame))] = 0
    r, c = meanFrame.shape
    thres = 30 #np.std( meanFrame )
    totalPixels = np.product( meanFrame.shape )
    for i, j in itertools.product(range(r), range(c)):
        n = i * c + j
        if n % 1000 == 0:
            print( '[INFO] Pixel %d/%d are done' % (n, totalPixels))
        u = np.mean(frames[:,i,j])
        v = np.std(frames[:,i,j])
        # Pixels which shave seen quite a lot of variation, set them to highest,
        # rest to zeros.
        #  print(v, end = ' ' )
        if u > 100 and v > thres:
            threshold[i,j] = 255
    #cv2.imwrite( os.path.join(datadir_, '%s.summary.mean.png' % infname), meanFrame )
    #cv2.imwrite( os.path.join(datadir_, '%s.threshold.png' % infname), threshold )
    #cv2.imwrite( os.path.join(datadir_,'%s.summary.std.png' % infname), (stdFrame + threshold)/2)
    return np.where(threshold == 255), [meanFrame, stdFrame, threshold] 

def get_time_slice( df, status ):
    f = df[df['status'] == status]['arduino_time'].values
    ts = [ dateutil.parser.parse(x) for x in f ]
    return ts[0], ts[-1]

def process( frames, threshold ):
    result = []
    for f in frames:
        signal = np.mean(f[threshold])
        fs = ''.join( [ chr(int(x)) for x in f[0,:] ]).rstrip().split(',')
        result.append( (dateutil.parser.parse(fs[0]), signal, fs) )
    return result

def plot_trial( data, frames, outfile ):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import io

    mpl.rcParams['text.usetex'] = False
    x, y, lines = zip(*data)

    cols = 'frame_time,arduino_time,arduino_signal,a,b,c,d,e,f,g,h,status,signal'.split(',')
    text = '\n'.join([','.join(x) for x in lines if len(x)==len(cols)])
    d = pd.read_csv( io.StringIO(text), sep = ',', names = cols )

    gridSize = (2, 3)
    ax1 = plt.subplot2grid( gridSize, (0,0), colspan = 1 )
    ax2 = plt.subplot2grid( gridSize, (0,1), colspan = 1 )
    ax3 = plt.subplot2grid( gridSize, (0,2), colspan = 1 )
    ax4 = plt.subplot2grid( gridSize, (1,0), colspan = 3 )

    ax1.imshow( frames[0] )
    ax1.set_title( 'Mean' )
    ax2.imshow( frames[1] )
    ax2.set_title( 'Std' )
    ax3.imshow( frames[2] )
    ax3.set_title( 'Pixels of interests' )

    ax4.plot( x, y )
    ax4.set_xlabel( 'Time' )

    ymin = np.min(y)
    for s in [ 'CS+', 'PUFF']:
        t0, t1 = get_time_slice(d, s)
        print( t0, t1 )
        ax4.plot( [t0, t1], [ ymin, ymin], lw = 3)
    ax4.set_ylabel( 'Signal' )

    plt.suptitle( 'Total frames %d' % len(x) )
    plt.savefig( outfile )

def run( infile_ ):
    global datadir_
    datadir_ = os.path.join( os.path.dirname( infile_ ), resdir_name_ )
    infilename = os.path.basename( infile_ )

    if not os.path.isdir( datadir_ ):
        os.makedirs( datadir_ )

    frames = read_all_frames( infile_ )
    picklefile = os.path.join(datadir_, '%s.threshold.pkl' % infile_)
    if not os.path.exists( picklefile ):
        res = preprocess_all_frames( frames )
        with open( picklefile, 'wb') as f:
            pickle.dump( res, f )
            print( "[INFO ] Wrote to picklefile %s" % picklefile )

    # Pickfile is found. load it.
    data = None
    with open( picklefile, 'rb') as f:
        threshold, summaryImg = pickle.load( f )
        data = process( frames, threshold)
        datafile = os.path.join(datadir_, '%s.data.csv' % infilename)
        with open(datafile, 'w' ) as f:
            for t, s, dataline in data:
                f.write( '%s %s\n' % (t, s))
        print( "[INFO ] Wrote to datafile %s" % datafile )
        plot_trial(data, summaryImg, os.path.join( datadir_, '%s.signal.png' % infilename) )

    return { 'data' : data }
    
def main():
    global infile_
    infile_ = sys.argv[1]
    run( infile_ )

if __name__ == '__main__':
    main()

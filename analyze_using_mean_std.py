#!/usr/bin/env python

import os
import sys
import tifffile
import numpy as np
import cv2
import itertools
import pickle
import dateutil.parser 

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
    thres = 20 #np.std( meanFrame )
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
    cv2.imwrite( os.path.join(datadir_, '%s.summary.mean.png' % infname), meanFrame )
    cv2.imwrite( os.path.join(datadir_, '%s.threshold.png' % infname), threshold )
    cv2.imwrite( os.path.join(datadir_,'%s.summary.std.png' % infname), (stdFrame + threshold)/2)
    return np.where(threshold == 255)

def process( frames, threshold ):
    result = []
    for f in frames:
        signal = np.mean(f[threshold])
        fs = ''.join( [ chr(x) for x in f[0,:] ]).rstrip().split(',')
        result.append( (dateutil.parser.parse(fs[0]),signal) )
    return result

def plot( data, outfile ):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['text.usetex'] = False
    x, y = zip(*data)
    plt.plot( x, y )
    plt.xlabel( 'Time' )
    plt.ylabel( 'Signal' )
    plt.savefig( outfile )

def main():
    global datadir_
    global infile_
    infile_ = sys.argv[1]
    datadir_ = os.path.join( os.path.dirname( infile_ ), resdir_name_ )
    infilename = os.path.basename( infile_ )

    if not os.path.isdir( datadir_ ):
        os.makedirs( datadir_ )

    frames = read_all_frames( infile )
    picklefile = os.path.join(datadir_, '%s.threshold.pkl' % infile_)
    if not os.path.exists( picklefile ):
        thres = preprocess_all_frames( frames )
        with open( picklefile, 'wb') as f:
            pickle.dump( thres, f )
            print( "[INFO ] Wrote to picklefile %s" % picklefile )

    # Pickfile is found. load it.
    with open( picklefile, 'rb') as f:
        threshold = pickle.load( f )
        data = process( frames, threshold)
        datafile = os.path.join(datadir_, '%s.data.csv' % infilename)
        with open(datafile, 'w' ) as f:
            for t, s in data:
                f.write( '%s %s\n' % (t, s))
        print( "[INFO ] Wrote to datafile %s" % datafile )
        plot(data, os.path.join( datadir_, '%s.signal.png' % infilename) )
    print( "[INFO ] All done." )
    

if __name__ == '__main__':
    main()

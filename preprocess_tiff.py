#!/usr/bin/env python

import sys
import tifffile
import numpy as np
import cv2
import itertools

cap_ = None

def fetch_a_good_frame( drop = 0, offset = 1 ):
    global cap_
    o = offset
    return next(cap_).asarray( )[o:-o,o:-o]

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
    meanFrame = np.mean( frames, axis = 0 )
    threshold = np.zeros_like(meanFrame)
    stdFrame = np.std( frames, axis = 0 )
    newframes = frames[:]

    #  newframes[np.where(frames < (meanFrame+stdFrame))] = 0
    r, c = meanFrame.shape
    thres = np.mean( meanFrame )
    totalPixels = np.product( meanFrame.shape )
    for i, j in itertools.product(range(r), range(c)):
        n = i * c + j
        if n % 1000 == 0:
            print( '[INFO] Pixel %d/%d are done' % (n, totalPixels))
        v = np.std(frames[:,i,j])
        # Pixels which shave seen quite a lot of variation, set them to highest,
        # rest to zeros.
        #  print(v, end = ' ' )
        if v > thres:
            threshold[i,j] = 255
    cv2.imwrite( 'summary.mean.png', meanFrame )
    cv2.imwrite( 'threshold.png', threshold )
    cv2.imwrite( 'summary.std.png', stdFrame )
    #  tifffile.imsave( 'processed.tif', data = newframes )
    #  print( 'Saved to processed.tif' )

def main():
    infile = sys.argv[1]
    outfile = None
    if len(sys.argv) > 2:
        outfile = sys.argv[2]
    frames = read_all_frames( infile )
    preprocess_all_frames( frames, outfile )
    

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import print_function 

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2018, Dilawar Singh"
__license__          = "GNU GPL"
__status__           = "Development"

import cv2
import math
from collections import defaultdict
import numpy as np
import tifffile

trajectory_ = [ ]
curr_loc_ = (100, 100)
static_features_ = defaultdict( int )
static_features_img_ = None
distance_threshold_ = 200
trajectory_file_ = None

# To keep track of template coordinates.
bbox_ = [ ]

frame_ = None # Current frame.
nframe_ = 0   # Index of currnet frame
fps_ = 1      # Frame per seocond

result_ = [ ]
frames_ = [ ]

# global window with callback function
window_ = "Mouse tracker"

# This is our template. Use must select it to begin with.
template_ = None
template_size_ = None

# MOG
fgbg_ = cv2.createBackgroundSubtractorMOG2()

# GMG
#  fgbg2_ = cv2.createBackgroundSubtractorKNN( )

fgbgkernel_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3) )
fgbg2_ = cv2.bgsegm.createBackgroundSubtractorGMG( )

def onmouse( event, x, y, flags, params ):
    global curr_loc_, frame_, window_ 
    global bbox_
    global template_, template_size_

    if template_ is None:
        # Draw Rectangle. Click and drag to next location then release.
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox_ = []
            bbox_ = [ x, y ]
        elif event == cv2.EVENT_LBUTTONUP:
            bbox_ += [x, y]

        if len( bbox_ ) == 4:
            print( 'bbox_ : %s' % str(bbox_))
            cv2.rectangle( frame_, (bbox_[0], bbox_[1]), (bbox_[2],bbox_[3]), 100, 2)
            x0,y0,x1,y1 = bbox_ 
            template_size_ = (y1-y0, x1-x0)
            template_ = frame_[y0:y1,x0:x1]
            cv2.imshow( window_, frame_ )

    # Else user is updating the current location of animal.
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            curr_loc_ = (x, y)
            # print( '[INFO] Current location updated to %s' % str( curr_loc_ ) )


def toGrey( frame ):
    if frame.max( ) < 255:
        return frame
    try:
        return cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    except Exception as e:
        return frame


def display_frame( frame, delay = 40 ):
    global window_ 
    try:
        cv2.imshow( window_, frame )
        cv2.waitKey( delay )
    except Exception as e:
        print( '[warn] could not display frame' )
        print( '\t Error was %s' % e )


def clip_frame( frame, box ):
    (r1, c1), (r2, c2 ) = box
    return frame[c1:c2,r1:r2]


def initialize_template( ):
    global window_, frame_
    global bbox_
    cv2.setMouseCallback(window_, onmouse)
    if template_ is None:
        while True:
            cv2.imshow( window_, frame_ )
            key = cv2.waitKey( 1 ) & 0xFF
            if key == ord( 'n' ):
                # print( '[INFO] Dropping this frame' )
                frame_ = fetch_a_good_frame( )
            elif key == ord( 'r' ):
                bbox_ = []
            elif key == ord( 'q' ):
                break

def initialize_global_window( ):
    global window_ 
    cv2.namedWindow( window_ )

def is_a_good_frame( frame ):
    if frame.max( ) < 100 or frame.min() > 150:
        # print( '[WARN] not a good frame: too bright or dark' )
        return False
    if frame.mean( ) < 50 or frame.mean() > 200:
        # print( '[WARN] not a good frame: not much variation' )
        return False
    return True

def readOrNext( cap ):
    try:
        return True, next(cap).asarray( )
    except Exception as e:
        print( e)
        return False, None

def fetch_a_good_frame( drop = 0 ):
    global cap_, bbox_
    global nframe_
    if bbox_:
        x0,y0,x1,y1 = bbox_ 
    ret, frame = readOrNext( cap_ )
    nframe_ += 1
    if bbox_:
        try:
            frame = frame[y0:y1,x0:x1] 
        except Exception as e:
            return None
    assert frame is not None, "Bad frame: %s" % frame
    return frame

def distance( p0, p1 ):
    x0, y0 = p0
    x1, y1 = p1
    return ((x0 - x1)**2 + (y0 - y1)**2) ** 0.5

def draw_point( frame, points, thickness = 2):
    for p in points:
        (x, y) = p.ravel()
        cv2.circle( frame, (x,y), 2, 30, thickness )
    return frame

def update_template( frame ):
    global curr_loc_ 
    global template_, template_size_
    h, w = template_size_ 
    c0, r0 = curr_loc_
    h = min( c0, r0, h, w)
    template_ = frame[ r0-h:r0+h, c0-h:c0+h ]
    cv2.imshow( 'template', template_ )
    cv2.waitKey( 1 )

def smooth( vec, N = 10 ):
    window = np.ones( N ) / N
    return np.convolve( vec, window, 'valid' )

def remove_fur( frame, kernelSize = 3 ):
    print( "[INFO ] Eroding -> Dilating image.  Making animal less furry!" )
    frame = cv2.morphologyEx( frame, cv2.MORPH_OPEN
            , np.ones((kernelSize,kernelSize), np.uint8)
            )
    return frame

def make_edges_dominant( frame, winsize=3 ):
    #  return cv2.GaussianBlur( frame, (winsize,winsize), 0 )
    return cv2.medianBlur(frame, winsize)

def compute_optical_flow( current, prev, blur = True, **kwargs ):

    base = np.zeros_like( current )
    flow = np.zeros_like(current)
    p0 = cv2.goodFeaturesToTrack( prev, 100, 0.1, 5)
    pNext = cv2.goodFeaturesToTrack( current, 100, 0.1, 5)

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev, current, p0, pNext
            , winSize = (11,51)
            , maxLevel = 2
            , criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            #  , flags = cv2.OPTFLOW_USE_INITIAL_FLOW
            )
    goodPrev = p0[st==1]
    goodNew = p1[st==1]
    for i, (old,new) in enumerate( zip(goodNew, goodPrev)):
        a, b = old.ravel()
        c, d = new.ravel()
        if distance((a,b), (c,d)) < 1:
            continue
        clr = 255
        flow = cv2.line(flow, (a,b), (c,d), clr, 2)
        flow = cv2.circle(flow,(a,b), 3, clr, 2)

    # Put good points.
    for p in p0:
        x, y = p[0]
        low = cv2.circle( base, (x,y), 1, 255, 1 )
    return flow, base

def use_background_subs( frame ):
    global fgbg_, fgbg2_
    frame = remove_fur( frame, 11 )
    motion = fgbg_.apply(frame)
    res = fgbg2_.apply( frame )
    res = cv2.morphologyEx( res, cv2.MORPH_OPEN, fgbgkernel_ )
    return res, motion


def compute_twitch( cur ):
    global curr_loc_ 
    global static_features_img_
    global trajectory_
    global result_

    if len(frames_) > 3:
        prev = frames_[-3]
    else:
        prev = frames_[-1]
    flow, other = compute_optical_flow(cur, prev)
    flow, other = use_background_subs(cur)
    result_.append( np.hstack((cur,other,flow)) )
    display_frame( result_[-1], 1 )


def process( args ):
    global cap_
    global box_
    global curr_loc_, frame_, fps_
    global nframe_
    global frame_, result_

    while True:
        if frame_ is None:
            print( "[WARN ] Could not find frame." )
            break
        frame_ = fetch_a_good_frame( ) 
        prev = frame_.copy()
        frames_.append( frame_ )
        nframe_ += 1
        compute_twitch( frame_ )
    print( '== All done' )

def main(args):
    # Extract video first
    global cap_, frame_, bbox_
    global trajectory_file_ 
    initialize_global_window( )
    print( 'Reading a tiff file' )
    cap_ = (x for x in tifffile.TiffFile( args.file ).pages )
    # Let user draw rectangle around animal on first frame.
    frame_ = fetch_a_good_frame( )
    if not args.bbox:
        initialize_template( )
    else:
        bbox_ = [ int(x) for x in args.bbox.split(',')]
    frame_ = fetch_a_good_frame( )
    process( args )

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Detect eye blinks in given recording.'''
    parser = argparse.ArgumentParser(description=description)
    class Args: pass 
    args = Args()
    parser.add_argument('--file', '-f'
        , required = True
        , help = 'Path of the video file or camera index. default camera 0'
        )
    parser.add_argument('--verbose', '-v'
        , required = False
        , action = 'store_true'
        , default = False
        , help = 'Show you whats going on?'
        )
    parser.add_argument('--bbox', '-b'
        , required = False
        , default = ''
        , type = str
        , help = 'Box to clip (csv) e.g 10,10,200,200 '
        )
    parser.parse_args(namespace=args)
    main( args )


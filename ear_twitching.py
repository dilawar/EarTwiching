#!/usr/bin/env python2
from __future__ import print_function 

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2018, Dilawar Singh"
__license__          = "GNU GPL"
__status__           = "Development"

import cv2
import math
from collections import defaultdict
import numpy as np
import gnuplotlib as gpl
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
        return True, cap.next( ).asarray( )
    except Exception as e:
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


def fix_current_location( frame ):
    """We have a hint of mouse location, now fix it by really locating the
    aninal
    """
    global curr_loc_, nframe_
    global template_
    global trajectory_
    try:
        update_template( frame )
        res = cv2.matchTemplate( frame, template_, cv2.TM_SQDIFF_NORMED )
        minv, maxv, (y,x), maxl = cv2.minMaxLoc( res )
        c0, r0 = curr_loc_
        w, h = template_.shape
        maxMatchPoint = (y+w/2, x+h/2)
        # cv2.circle( frame, curr_loc_, 5, 100, 5)
        curr_loc_ = maxMatchPoint
        cv2.circle( frame, curr_loc_, 10, 255, 3)
        trajectory_.append( curr_loc_ )
        print( '- Time %.2f, Current loc %s', ( nframe_/fps_, str(curr_loc_)))
        time = nframe_ / float( fps_ )
        # Append to trajectory file.
        with open( trajectory_file_, 'a' ) as trajF:
            c0, r0 = curr_loc_
            trajF.write( '%g %d %d\n' % (time, c0, r0) )

    except Exception as e:
        print( 'Failed with %s' % e )
        return 


def update_mouse_location( points, frame ):
    global curr_loc_
    global static_features_img_
    global distance_threshold_

    c0, r0 = curr_loc_ 
    res = {}
    newPoints = [ ]
    if points is None:
        return None, None
    sumC, sumR = 0.0, 0.0

    for p in points:
        (x,y) = p.ravel( )
        x, y = int(x), int(y)

        # We don't want points which are far away from current location.
        if distance( (x,y), curr_loc_ ) > distance_threshold_:
            continue 

        # if this point is in one of static feature point, reject it
        if static_features_img_[ y, x ] > 1.5:
            continue
        newPoints.append( (x,y) )
        sumR += y
        sumC += x

    newPoints = np.array( newPoints )
    ellipse = None
    try:
        if( len(newPoints) > 5 ):
            ellipse = cv2.fitEllipse( newPoints )
    except Exception as e:
        pass
    if len( newPoints ) > 0:
        curr_loc_ = ( int(sumC / len( newPoints )), int(sumR / len( newPoints)) )
        

    ## Fix the current location
    fix_current_location( frame )
    
    res[ 'ellipse' ] = ellipse 
    res[ 'contour' ] = newPoints

    return res

def insert_int_corners( points ):
    """Insert or update feature points into an image by increasing the pixal
    value by 1. If a feature point is static, its count will increase
    drastically.
    """
    global static_features_img_
    global distance_threshold_
    if points is None:
        return 
    for p in points:
        (x,y) = p.ravel()
        static_features_img_[int(y),int(x)] += 1

def smooth( vec, N = 10 ):
    window = np.ones( N ) / N
    return np.correlate( vec, window, 'valid' )

def remove_fur( frame ):
    kernelSize = 3
    print( "[INFO ] Eroding -> Dilating image.  Making animal less furry!" )
    frame = cv2.morphologyEx( frame, cv2.MORPH_OPEN
            , np.ones((kernelSize,kernelSize), np.uint8)
            )
    return frame

def compute_optical_flow( current, prev, blur = True, **kwargs ):
    flow = np.zeros_like(current)
    p0 = cv2.goodFeaturesToTrack( prev, 100, 0.1, 10)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev, current, p0, None
            , winSize = (17,17)
            , maxLevel = 1
            , criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
    goodPrev = p0[st==1]
    goodNew = p1[st==1]
    for i, (old,new) in enumerate( zip(goodNew, goodPrev)):
        a, b = old.ravel()
        c, d = new.ravel()
        clr = np.random.randint( 100, 255 )
        flow = cv2.line(flow, (a,b), (c,d), clr, 2)
        flow = cv2.circle(flow,(a,b), 3, clr, 2)
    return flow


def compute_twitch( cur, prev ):
    global curr_loc_ 
    global static_features_img_
    global trajectory_
    global result_
    flow = compute_optical_flow(cur, prev)
    result_.append( np.hstack((cur,flow)) )
    display_frame( result_[-1], 1 )


def process( args ):
    global cap_
    global box_
    global curr_loc_, frame_, fps_
    global nframe_
    global frame_, result_

    while True:
        prev = frame_[:]
        frame_ = fetch_a_good_frame( ) 
        frame_ = remove_fur( frame_ )
        frames_.append( frame_ )
        nframe_ += 1
        if frame_ is None:
            break
        try:
            compute_twitch( frame_, frames_[-3] )
        except Exception as e:
            print("[WARN ] Failed to compute twich. Error was %s" % e)
            
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


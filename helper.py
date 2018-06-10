"""helper.py: 

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import pandas as pd
import numpy as np
import io

def lines_to_dataframe( lines ):
    cols = 't1,t2,s1,a,b,c,d,e,f,g,h,status,sig1,sig2'.split(',')
    d = pd.read_csv( io.StringIO(lines), sep = ',', names = cols
            , parse_dates = [ 't2', 't1'] )
    # Drop invlid lines.
    d = d.dropna()
    return d 

def get_time_slice( df, status ):
    f = df[df['status'] == status]['t1'].values
    if len(f) > 2:
        return f[0], f[-1]
    return 0, 0

def _max(a, b):
    if a is None:
        return b
    if b is None:
        return a 
    return max(a,b)

def _min(a, b):
    if a is None:
        return b
    if b is None:
        return a 
    return min(a,b)

def _interp( x, x0, y0 ):
    return np.interp(x, x0, y0)

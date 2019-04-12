#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:43:46 2019

@author: maximoskaliakatsos-papakostas
"""

import score2np as s2n
import os
cwd = os.getcwd()
import pickle
import numpy as np

folderName = cwd + '/bc404'
parts_for_surface = 'all'
time_res = 16
ts_num = 4
ts_den = 4

m, l = s2n.get_time_sig_parts_np_from_folder(folderName, parts_for_surface, time_res, ts_num, ts_den, print_progress=True, transpose=True, range_trim=True)

# segments: array with 64-tuples of columns
# segments = []
serialised_segments = []
start_idx = 0
rows = m.shape[0]
columns = 64
step = 16
for i in range( int(m.shape[1]/step) ):
    end_idx = start_idx + columns
    if end_idx < m.shape[1]:
        # segments.append( m[ : , start_idx:end_idx ] )
        x = m[ : , start_idx:end_idx ]
        serialised_segments.append( np.reshape( x, (1 , rows*columns ) ) )
        start_idx += step

serialised_segments = np.vstack( serialised_segments )
serialised_segments = np.array( serialised_segments, dtype=np.float32 )

# pickle data in two parts
with open('saved_data' + os.sep + 'data_tower.pickle', 'wb') as handle:
    d = {}
    d['serialised_segments'] = serialised_segments
    d['rows'] = rows
    d['columns'] = columns
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
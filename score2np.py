#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:02:22 2018

@author: maximoskaliakatsos-papakostas
"""

from music21 import *
import numpy as np
import os
import glob

# the user should give the file name - the folder with the files
def get_parts_np_from_file(fileName, parts_for_surface, time_res):
    # INPUTS:
    # fileName - string: the name of the file - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # OUTPUTS:
    # m: the score matrix in np array
    # m_length: the number of columns in the matrix
    
    p = converter.parse(fileName)
    
    # tmp list that will include the lists of all pitches
    tmp_all_pitches = []
    
    if parts_for_surface == 'all':
        parts_for_surface = range(len(p.parts))
    
    # make surface matrix
    for i in parts_for_surface:
        # get part
        tmp_part = p.parts[i]
        # tmp list that will include the lists of all pitches in the part
        tmp_part_pitches = []
        # get array of measures
        measures = [m for m in tmp_part.getElementsByClass('Measure')]
        # for all measures
        for m in measures:
            # get time signature of measure
            ts = m.flat.timeSignature
            if ts != None:
                # tmp_ts = ts[0]
                if ts.numerator == 4 and ts.denominator == 4:
                    measureLength = time_res
                elif ts.numerator == 3 and ts.denominator == 4:
                    measureLength = int(3.0*time_res/4.0)
                elif ts.numerator == 3 and ts.denominator == 8:
                    measureLength = int(3.0*time_res/8.0)
                elif ts.numerator == 3 and ts.denominator == 1:
                    measureLength = int(3.0*time_res)
                elif ts.numerator == 12 and ts.denominator == 8:
                    measureLength = int(12.0*time_res/8.0)
                elif ts.numerator == 3 and ts.denominator == 2:
                    measureLength = int(3.0*time_res/2.0)
                else:
                    print("unknown time signature: ", ts.numerator, ts.denominator)
            notes = m.flat.notes
            # tmp list that stores the pitches for the measure
            tmp_measure_pitches = np.zeros((128, measureLength))
            for n in notes:
                offset_value = int( eval(str(n.offset))*time_res/4.0 )
                duration_value = int(n.duration.quarterLength*time_res/4.0)
                if n.isChord:
                    for nn in n.pitches:
                        midi_number = nn.midi
                        # print(midi_number, " - ", offset_value, " - ", duration_value)
                        tmp_measure_pitches[midi_number, offset_value] = duration_value
                else:
                    midi_number = n.pitch.midi
                    tmp_measure_pitches[midi_number, offset_value] = duration_value
            
            # tmp_all_pitches.append(tmp_measure_pitches)
            if len(tmp_part_pitches) == 0:
                tmp_part_pitches = np.array(tmp_measure_pitches)
            else:
                tmp_part_pitches = np.hstack((tmp_part_pitches, tmp_measure_pitches))
        
        tmp_all_pitches.append(tmp_part_pitches)
        
        all_pitches = np.array(tmp_all_pitches[0])
        for a in tmp_all_pitches:
            all_pitches[ all_pitches == 0 ] = a[ all_pitches == 0 ]
        
    return all_pitches, np.size(all_pitches, axis=1)

def get_time_sig_parts_np_from_file(fileName, parts_for_surface, time_res, ts_num, ts_den):
    # INPUTS:
    # fileName - string: the name of the file - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # OUTPUTS:
    # m: the score matrix in np array
    # m_length: the number of columns in the matrix
    
    p = converter.parse(fileName)
    
    # tmp list that will include the lists of all pitches
    tmp_all_pitches = []
    all_pitches = []
    all_pitches_size = []
    if parts_for_surface == 'all':
        parts_for_surface = range(len(p.parts))
    
    # make surface matrix
    for i in parts_for_surface:
        # get part
        tmp_part = p.parts[i]
        # tmp list that will include the lists of all pitches in the part
        tmp_part_pitches = []
        # get array of measures
        measures = [m for m in tmp_part.getElementsByClass('Measure')]
        # assume a none ts
        ts = None
        # for all measures
        for m in measures:
            # get time signature of measure
            new_ts = m.flat.timeSignature
            if new_ts != None:
                ts = new_ts
            if ts != None:
                if ts.numerator == ts_num and ts.denominator == ts_den:
                    # tmp_ts = ts[0]
                    if ts.numerator == 4 and ts.denominator == 4:
                        measureLength = time_res
                    elif ts.numerator == 3 and ts.denominator == 4:
                        measureLength = int(3.0*time_res/4.0)
                    elif ts.numerator == 3 and ts.denominator == 8:
                        measureLength = int(3.0*time_res/8.0)
                    elif ts.numerator == 3 and ts.denominator == 1:
                        measureLength = int(3.0*time_res)
                    elif ts.numerator == 12 and ts.denominator == 8:
                        measureLength = int(12.0*time_res/8.0)
                    elif ts.numerator == 3 and ts.denominator == 2:
                        measureLength = int(3.0*time_res/2.0)
                    else:
                        print("unknown time signature: ", ts.numerator, ts.denominator)
                    notes = m.flat.notes
                    # tmp list that stores the pitches for the measure
                    tmp_measure_pitches = np.zeros((128, measureLength))
                    for n in notes:
                        offset_value = int( eval(str(n.offset))*time_res/4.0 )
                        duration_value = int(n.duration.quarterLength*time_res/4.0)
                        if n.isChord:
                            for nn in n.pitches:
                                midi_number = nn.midi
                                # print(midi_number, " - ", offset_value, " - ", duration_value)
                                tmp_measure_pitches[midi_number, offset_value] = duration_value
                        else:
                            midi_number = n.pitch.midi
                            tmp_measure_pitches[midi_number, offset_value] = duration_value
                    # tmp_all_pitches.append(tmp_measure_pitches)
                    if len(tmp_part_pitches) == 0:
                        tmp_part_pitches = np.array(tmp_measure_pitches)
                    else:
                        tmp_part_pitches = np.hstack((tmp_part_pitches, tmp_measure_pitches))
        if len(tmp_part_pitches) > 0:
            tmp_all_pitches.append(tmp_part_pitches)
    if len(tmp_all_pitches) > 0:
        all_pitches = np.array(tmp_all_pitches[0])
        for a in tmp_all_pitches:
            all_pitches[ all_pitches == 0 ] = a[ all_pitches == 0 ]
        all_pitches_size = np.size(all_pitches, axis=1)
        
    return all_pitches, all_pitches_size

def get_rel_pcp_np_from_file(fileName, parts_for_surface, time_res):
    # INPUTS:
    # fileName - string: the name of the file - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # OUTPUTS:
    # m: the score matrix in np array
    # m_length: the number of columns in the matrix
    
    p = converter.parse(fileName)
    
    # tmp list that will include the lists of all pitches
    tmp_all_pitches = []
    
    if parts_for_surface == 'all':
        parts_for_surface = range(len(p.parts))
    
    # make surface matrix
    for i in parts_for_surface:
        # get part
        tmp_part = p.parts[i]
        # tmp list that will include the lists of all pitches in the part
        tmp_part_pitches = []
        # get array of measures
        measures = [m for m in tmp_part.getElementsByClass('Measure')]
        # for all measures
        for m in measures:
            # get time signature of measure
            ts = m.flat.timeSignature
            if ts != None:
                # tmp_ts = ts[0]
                if ts.numerator == 4 and ts.denominator == 4:
                    measureLength = time_res
                elif ts.numerator == 3 and ts.denominator == 4:
                    measureLength = int(3.0*time_res/4.0)
                elif ts.numerator == 3 and ts.denominator == 8:
                    measureLength = int(3.0*time_res/8.0)
                else:
                    print("unknown time signature: ", ts.numerator, ts.denominator)
            notes = m.flat.notes
            # tmp list that stores the pitches for the measure
            tmp_measure_pitches = np.zeros((128, measureLength))
            for n in notes:
                offset_value = int( eval(str(n.offset))*time_res/4.0 )
                duration_value = int(n.duration.quarterLength*time_res/4.0)
                if n.isChord:
                    for nn in n.pitches:
                        midi_number = nn.midi
                        # print(midi_number, " - ", offset_value, " - ", duration_value)
                        tmp_measure_pitches[midi_number, offset_value] = duration_value
                else:
                    midi_number = n.pitch.midi
                    tmp_measure_pitches[midi_number, offset_value] = duration_value
            
            # tmp_all_pitches.append(tmp_measure_pitches)
            if len(tmp_part_pitches) == 0:
                tmp_part_pitches = np.array(tmp_measure_pitches)
            else:
                tmp_part_pitches = np.hstack((tmp_part_pitches, tmp_measure_pitches))
        
        tmp_all_pitches.append(tmp_part_pitches)
        
        all_pitches = np.array(tmp_all_pitches[0])
        for a in tmp_all_pitches:
            all_pitches[ all_pitches == 0 ] = a[ all_pitches == 0 ]
        
    return all_pitches, np.size(all_pitches, axis=1)
# end

# running for all files in folder
def get_parts_3Dnp_from_folder(folderName, parts_for_surface, time_res):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # OUTPUTS:
    # m: the score matrices in np array (n_pieces, 128, max_len)
    # m_length: the number of columns in each matrix (n_pieces, )
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    
    for fileName in allDocs:
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    # pad-em-all
    for i in range(len(all_matrices)):
        m = all_matrices[i]
        # check if padding needed
        if m.shape[1] < max_length:
            # make a padding matrix
            padder = -1.0*np.ones((128, max_length-m.shape[1]))
            all_matrices[i] = np.hstack( (m, padder) )
    
    all_matrices = np.array(all_matrices)
    return all_matrices, all_lengths

'''
# running for all files in folder but returns all pieces concatenated in a large matrix
def get_concat_parts_np_from_folder(folderName, parts_for_surface, time_res, transpose = False):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    
    for fileName in allDocs:
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    # pad-em-all
    for i in range(len(all_matrices)):
        m = all_matrices[i]
        # check if padding needed
        if m.shape[1] < max_length:
            # make a padding matrix
            padder = -1.0*np.ones((128, max_length-m.shape[1]))
            all_matrices[i] = np.hstack( (m, padder) )
    
    all_matrices = np.hstack(all_matrices)
    
    # check if +-6 transposition has been asked
    if transpose:
        initMat = np.array(all_matrices)
        for i in range(1,7):
            tmpMat = np.roll(initMat, i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
        for i in range(1,7):
            tmpMat = np.roll(initMat, -i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
    
    return all_matrices
'''

def compute_features(m):
    horizontal_range = 4
    # initially 2 features: horizontal and vertical density
    f = np.zeros( (2, m.shape[1]) )
    for i in range(horizontal_range, m.shape[1]-horizontal_range):
        tmp_1 = np.sum( m[:, (i-horizontal_range):(i+horizontal_range) ] , axis=0) >= 1
        if len(tmp_1) == 0:
            tmp_1 = [0]
        f[0,i] = np.mean( tmp_1 )
        # smoothening
        # f[0,i] = np.mean(f[0, (i-horizontal_range):(i+horizontal_range)])
        tmpMat = m[:, (i-horizontal_range):(i+horizontal_range) ]
        tmpMat_part = tmpMat[:, np.sum( tmpMat , axis=0) >= 1]
        tmp_2 = np.sum(tmpMat_part, axis=0)/4.0
        if len(tmp_2) == 0:
            tmp_2 = [0]
        f[1,i] = np.mean( tmp_2 )
        # smoothening
        # f[1,i] = np.mean(f[1, (i-horizontal_range):(i+horizontal_range)])
    # moving average
    f[0,:] = np.convolve(f[0,:], np.ones(horizontal_range)/horizontal_range, mode="same")
    f[1,:] = np.convolve(f[1,:], np.ones(horizontal_range)/horizontal_range, mode="same")
    return f
# end compute_features

# running for all files in folder but returns all pieces concatenated in a large matrix
def get_concat_parts_np_from_folder(folderName, parts_for_surface, time_res, transpose = False, bin_out=True, voice_aug=False, sparse_aug=False, padding=False, print_progress=False):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    # f: the feature matrix in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    
    for idx, fileName in enumerate(allDocs):
        if print_progress:
            print(str(idx) + ' - processsing: ', fileName)
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    # pad-em-all
    if padding:
        for i in range(len(all_matrices)):
            m = all_matrices[i]
            # check if padding needed
            if m.shape[1] < max_length:
                # make a padding matrix
                padder = -1.0*np.ones((128, max_length-m.shape[1]))
                all_matrices[i] = np.hstack( (m, padder) )
    
    all_matrices = np.hstack(all_matrices)
    
    # check if +-6 transposition has been asked
    if transpose:
        print('performing transpositions')
        initMat = np.array(all_matrices)
        for i in range(1,7):
            print('transposition: ', str(i))
            tmpMat = np.roll(initMat, i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
        for i in range(1,7):
            print('transposition: ', str(-i))
            tmpMat = np.roll(initMat, -i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
    
    # check if only binary output is required
    if bin_out:
        print('generating binary output')
        all_matrices[all_matrices > 0] = 1
    
    # check if voice-limit augmentation has been asked
    if voice_aug:
        # do voice augmentation
        print('performing voice augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) > 3:
                passes = all_matrices[:,i].argsort()[-2:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
            elif np.count_nonzero(all_matrices[:,i]) >= 1:
                passes = all_matrices[:,i].argsort()[-1:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # check if sparsity augmentation has been asked
    if sparse_aug:
        # do sparsity augmentation
        print('performing sparsity augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) >= 1:
                # decide for passing the column
                if np.random.rand(1)[0] < 0.3:
                    new_mat[:,i] = all_matrices[:,i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # compute feature matrix
    # print('computing feature matrix')
    # f = compute_features(all_matrices)
    '''
    horizontal_range = 8
    # initially 2 features: horizontal and vertical density
    f = np.zeros( (2, all_matrices.shape[1]) )
    for i in range(horizontal_range, all_matrices.shape[1]-horizontal_range):
        f[0,i] = np.mean( np.sum( all_matrices[:, (i-horizontal_range):(i+horizontal_range) ] , axis=0) >= 1 )
        tmpMat = all_matrices[:, (i-horizontal_range):(i+horizontal_range) ]
        tmpMat_part = tmpMat[:, np.sum( tmpMat , axis=0) >= 1]
        f[1,i] = np.mean( np.sum(tmpMat_part, axis=0)/4.0 )
    '''
    
    return all_matrices

# running for all files in folder but returns all pieces concatenated in a large matrix
def get_concat_rel_pcp_np_from_folder(folderName, parts_for_surface, time_res, parts_for_tonality, bin_out=True):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    # f: the feature matrix in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    # keep all tonalities
    all_tonalities = [];
    
    for fileName in allDocs:
        print("processing: ", fileName)
        m, l = get_parts_np_from_file(fileName, parts_for_surface, time_res)
        all_matrices.append(m)
        all_lengths.append(l)
        # get tonalities
        t, lt = get_parts_np_from_file(fileName, parts_for_tonality, time_res)
        all_tonalities.append(t)
    
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    '''
    # pad-em-all
    if padding:
        for i in range(len(all_matrices)):
            m = all_matrices[i]
            # check if padding needed
            if m.shape[1] < max_length:
                # make a padding matrix
                padder = -1.0*np.ones((128, max_length-m.shape[1]))
                all_matrices[i] = np.hstack( (m, padder) )
    '''
    
    all_matrices = np.hstack(all_matrices)
    all_tonalities = np.hstack(all_tonalities)
    
    '''
    # check if +-6 transposition has been asked
    if transpose:
        print('performing transpositions')
        initMat = np.array(all_matrices)
        for i in range(1,7):
            print('transposition: ', str(i))
            tmpMat = np.roll(initMat, i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
        for i in range(1,7):
            print('transposition: ', str(-i))
            tmpMat = np.roll(initMat, -i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
    '''
    # check if only binary output is required
    if bin_out:
        print('generating binary output')
        all_matrices[all_matrices > 0] = 1
        all_tonalities[all_tonalities > 0] = 1
    
    '''
    # check if voice-limit augmentation has been asked
    if voice_aug:
        # do voice augmentation
        print('performing voice augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) > 3:
                passes = all_matrices[:,i].argsort()[-2:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
            elif np.count_nonzero(all_matrices[:,i]) >= 1:
                passes = all_matrices[:,i].argsort()[-1:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # check if sparsity augmentation has been asked
    if sparse_aug:
        # do sparsity augmentation
        print('performing sparsity augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) >= 1:
                # decide for passing the column
                if np.random.rand(1)[0] < 0.3:
                    new_mat[:,i] = all_matrices[:,i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # compute feature matrix
    print('computing feature matrix')
    f = compute_features(all_matrices)
    
    horizontal_range = 8
    # initially 2 features: horizontal and vertical density
    f = np.zeros( (2, all_matrices.shape[1]) )
    for i in range(horizontal_range, all_matrices.shape[1]-horizontal_range):
        f[0,i] = np.mean( np.sum( all_matrices[:, (i-horizontal_range):(i+horizontal_range) ] , axis=0) >= 1 )
        tmpMat = all_matrices[:, (i-horizontal_range):(i+horizontal_range) ]
        tmpMat_part = tmpMat[:, np.sum( tmpMat , axis=0) >= 1]
        f[1,i] = np.mean( np.sum(tmpMat_part, axis=0)/4.0 )
    '''
    
    # tune to tonality
    curr_base = 0

    # keep also initial composition
    initial_mat = np.array(all_matrices);
    
    for i in range(all_matrices.shape[1]):
        if np.count_nonzero( all_tonalities[:,i] ) > 0:
            curr_base = np.min(np.nonzero(all_tonalities[:,i]))%12
        all_matrices[:,i] = np.roll( all_matrices[:,i], -curr_base )
        
    # modulo-12 them
    all_matrices = np.vstack( ( all_matrices, np.zeros( (4, all_matrices.shape[1]) ) ) )
    pcp = np.zeros((12, all_matrices.shape[1]))
    for i in range(0,all_matrices.shape[0], 12):
        pcp += all_matrices[i:min( [(i+12), all_matrices.shape[0]] ), :]
    
    return pcp, all_tonalities, initial_mat

def get_time_sig_parts_np_from_folder(folderName, parts_for_surface, time_res, ts_num, ts_den, range_trim=False, transpose=False, bin_out=True, voice_aug=False, sparse_aug=False, padding=False, print_progress=False):
    # INPUTS:
    # folderName - string: the name of the folder - full path
    # parts_for_surface - int array or 'all': parts to include to score
    # time_res - int: e.g. 16 or 32 etc.
    # transpose: True/False - augment or not with all +-6 transpositions
    # OUTPUTS:
    # m: the score matrices in np array (128, max_len*n_pieces)
    # f: the feature matrix in np array (128, max_len*n_pieces)
    
    allDocs = glob.glob(folderName + os.sep + "*.xml")
    
    # keep all matrices for each file
    all_matrices = []
    # keep the respective lengths to define -1 - padding
    all_lengths = []
    
    for idx, fileName in enumerate(allDocs):
        if print_progress:
            print(str(idx) + ' - processsing: ', fileName)
        m, l = get_time_sig_parts_np_from_file(fileName, parts_for_surface, time_res, ts_num, ts_den)
        if len(m) > 0:
            all_matrices.append(m)
            all_lengths.append(l)
    if len(all_matrices) == 0:
        print('NO PIECE WITH the requested time sig - possibly an error will occur. See score2np.py line 601')
    # find max length for padding
    all_lengths = np.array(all_lengths)
    max_length = np.max(all_lengths)
    
    # pad-em-all
    if padding:
        for i in range(len(all_matrices)):
            m = all_matrices[i]
            # check if padding needed
            if m.shape[1] < max_length:
                # make a padding matrix
                padder = -1.0*np.ones((128, max_length-m.shape[1]))
                all_matrices[i] = np.hstack( (m, padder) )
                all_lengths[i] = all_lengths[i] + max_length-m.shape[1]
    
    all_matrices = np.hstack(all_matrices)
    
    # check if +-6 transposition has been asked
    if transpose:
        print('performing transpositions')
        initMat = np.array(all_matrices)
        for i in range(1,7):
            print('transposition: ', str(i))
            tmpMat = np.roll(initMat, i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
        for i in range(1,7):
            print('transposition: ', str(-i))
            tmpMat = np.roll(initMat, -i, axis=0)
            all_matrices = np.hstack( (all_matrices, tmpMat) )
    
    # check if only binary output is required
    if bin_out:
        print('generating binary output')
        all_matrices[all_matrices > 0] = 1
    
    # check if voice-limit augmentation has been asked
    if voice_aug:
        # do voice augmentation
        print('performing voice augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) > 3:
                passes = all_matrices[:,i].argsort()[-2:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
            elif np.count_nonzero(all_matrices[:,i]) >= 1:
                passes = all_matrices[:,i].argsort()[-1:][::1]
                new_mat[passes, i] = all_matrices[passes, i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    # check if sparsity augmentation has been asked
    if sparse_aug:
        # do sparsity augmentation
        print('performing sparsity augmentation')
        new_mat = np.zeros( all_matrices.shape )
        for i in range(all_matrices.shape[1]):
            if np.count_nonzero(all_matrices[:,i]) >= 1:
                # decide for passing the column
                if np.random.rand(1)[0] < 0.3:
                    new_mat[:,i] = all_matrices[:,i]
        all_matrices = np.hstack( (all_matrices, new_mat) )
    
    if range_trim:
        h_sum = np.sum(all_matrices, axis=1)
        max_pitch = np.max( np.where( h_sum > 0 )[0] )
        min_pitch = np.min( np.where( h_sum > 0 )[0] )
        print('min_pitch: ', min_pitch)
        print('max_pitch: ', max_pitch)
        all_matrices = all_matrices[ min_pitch:(max_pitch+1) , : ]
    
    return all_matrices, all_lengths
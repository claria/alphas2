#!/usr/bin/env python2

import sys, os
import numpy as np
import math

def main(args):
    data_file = "InclusiveJets_Table_postCWR.txt"
    stat_correlations_file = "InclusiveJets_StatCorrelations_postCWR.txt"
    
    in_data = np.genfromtxt(data_file,dtype=float,skip_header=0,names=True)
    in_stat_correlations = np.genfromtxt(stat_correlations_file,dtype = float)

    np_cors = np.genfromtxt('NPCorrection_2011_combinedBornKt.txt',dtype = float, names=True)
    ew_cors = np.genfromtxt('EWK.dat', dtype=float, names=True)

    data = dict()
    for arr in in_data.dtype.names:
        if ('_lo' in arr) and (not arr in ['pt_lo', 'y_lo']) :
            data[arr] = in_data[arr] * -1.
        else:
            data[arr] = in_data[arr]




    data['SINGLE_PION'] = ((data['xs'] * data['SINGLE_PION_up']) - (data['xs'] * data['SINGLE_PION_lo'])) /2.0
    del data['SINGLE_PION_up']
    del data['SINGLE_PION_lo']
    print data.keys()
    print data['y_lo']
    print data['SINGLE_PION']
    
    data['SINGLE_PION_BAR'] = data['SINGLE_PION'].copy() / math.sqrt(2)
    data['SINGLE_PION_BAR'][data['y_lo'] >  1.0] = 0.0
    data['SINGLE_PION_END'] = data['SINGLE_PION'].copy()
    data['SINGLE_PION_END'][data['y_lo'] <  1.5] = 0.0
    data['SINGLE_PION_0005'] = data['SINGLE_PION'].copy() / math.sqrt(2)
    data['SINGLE_PION_0005'][data['y_lo'] !=  0.0] = 0.0
    data['SINGLE_PION_0510'] = data['SINGLE_PION'].copy() / math.sqrt(2)
    data['SINGLE_PION_0510'][data['y_lo'] !=  0.5] = 0.0
    data['SINGLE_PION_1015'] = data['SINGLE_PION'].copy() / math.sqrt(2)
    data['SINGLE_PION_1015'][data['y_lo'] !=  1.0] = 0.0
    del data['SINGLE_PION']


    data['ewcor'] = (1. + ew_cors['delta_tree']) * (1. + ew_cors['delta_loop'])
    data['npcor'] = np_cors['TOT_NP']
    data['npcorerr'] = np_cors['TOT_NP_stat'] * in_data['xs']


    #np.savez(outfile, **data)
    # conatenate arrays
    import operator
    from collections import OrderedDict
    res = OrderedDict(sorted(data.iteritems(), key=operator.itemgetter(0)))
    print res.keys()
    res_arr = res.values()
    res_arr = (np.vstack(res_arr)).transpose()
    header = '  '.join(['{:>12s}'.format(k) for k in res.keys()])
    np.savetxt('CMS_Inclusive_Jets_2011_prep.txt', res_arr, header=header, fmt='%20.5g')
    
def get_max_from_abs(np1,np2):
    res = np.zeros(np1.shape)
    for i in range(0,len(np1)):
        res[i] = np1[i] if (np.abs(np1[i]) > np.abs(np2[i])) else np2[i]
    return res

def get_cor_matrix_from_table(correlation_table, bin_1, bin_2):
    cor_matrix = np.zeros([len(bin_1),len(bin_1)])
    for i in range(0,len(bin_1)):
        for j in range(0,len(bin_1)):
            #print bin_1[i],bin_2[j],bin_1[j],bin_2[i]
            if bin_1[i] == bin_1[j]:
                corr_factor = [item[4] for item in correlation_table \
                        if (item[0] == bin_1[j] and item[1] == bin_2[i] \
                        and item[2] == bin_1[i] and item[3] == bin_2[j])][0]
                cor_matrix[i][j] = corr_factor
    return cor_matrix
    
#def get_cor_matrix_from_syst(data)
#    

def get_cov_from_stat(correlation_matrix, stat):
    cov_matrix = np.zeros([len(stat),len(stat)])
    for i in range(0,len(stat)):
        for j in range(0,len(stat)):
            cov_matrix[i][j] = correlation_matrix[i][j]*stat[i]*stat[j] 
    return cov_matrix


def get_cor_from_syst(syst):
    cor_matrix = np.zeros([len(syst),len(syst)])
    for i in range(0,len(syst)):
        for j in range(0,len(syst)):
            cor_matrix[i][j] = syst[i] * syst[j]
    return cor_matrix

def get_cor_from_uncor(uncor):
    cor_matrix = np.zeros([len(uncor),len(uncor)])
    for i in range(0,len(uncor)):
        cor_matrix[i][i] = uncor[i]*uncor[i]
    return cor_matrix

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

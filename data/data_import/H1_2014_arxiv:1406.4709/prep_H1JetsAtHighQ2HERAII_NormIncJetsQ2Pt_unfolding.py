#!/usr/bin/env python2

import os, sys
import numpy as np
import math

def main(args):

    data_file = "H1JetsAtHighQ2HERAII_NormIncJetsQ2Pt_unfolding_raw.txt"
    

    in_data = np.genfromtxt(data_file,dtype=float,skip_header=0,names=True)

    data = dict()
    for arr in in_data.dtype.names:
        data[arr] = in_data[arr]

    print data.keys()

    # Split JES
    data['JESu_lo'] = (data['JES_lo'] * np.sqrt(2.)) * 0.5
    data['JESu_up'] = (data['JES_up'] * np.sqrt(2.)) * 0.5
    data['JESc_lo'] = (data['JES_lo'] * np.sqrt(2.)) * 0.5
    data['JESc_up'] = (data['JES_up'] * np.sqrt(2.)) * 0.5

    del data['JES_lo']
    del data['JES_up']

    #Split RCES
    data['EHFSu_lo'] = (data['EHFS_lo'] * np.sqrt(2.)) * 0.5
    data['EHFSu_up'] = (data['EHFS_up'] * np.sqrt(2.)) * 0.5
    data['EHFSc_lo'] = (data['EHFS_lo'] * np.sqrt(2.)) * 0.5
    data['EHFSc_up'] = (data['EHFS_up'] * np.sqrt(2.)) * 0.5

    del data['EHFS_lo']
    del data['EHFS_up']

    #Split Model
    data['Modu_lo'] = (data['Mod_lo'] * np.sqrt(2.)) * 0.75
    data['Modu_up'] = (data['Mod_up'] * np.sqrt(2.)) * 0.75
    data['Modc_lo'] = (data['Mod_lo'] * np.sqrt(2.)) * 0.25
    data['Modc_up'] = (data['Mod_up'] * np.sqrt(2.)) * 0.25

    del data['Mod_lo']
    del data['Mod_up']

    #Decorrelate between Q2 bins (split up in separate sources)
    q2bins = sorted(list(set(data['Q2_lo'])))
    for source in ['Ee_up', 'Ee_lo', 'ThE_up', 'ThE_lo', 'IDe_up', 'IDe_lo', 'Modc_up', 'Modc_lo']:
        for n, q2bin in enumerate(q2bins):
            label = '{}_{}_{}'.format(source.split('_', 1)[0], n, source.split('_')[1])
            data[label] = data[source].copy()
            data[label][data['Q2_lo'] != q2bin] = 0.0
        del data[source]
 

    #ordered_list = ['ylow', 'yhigh', 'ptlow', 'pthigh', 'xs', 'stat', 'uncor', 'High_PT_EXTR', 'FLAVOUR', 'RelativeStatEC2', 'RelativeJERCHF', 'SINGLE_PION_0005', 'RelativeJERC2', 'ABSOLUTE', 'PileUpJetRate', 'unf_err', 'PileUpPt', 'npcor', 'SINGLE_PION_BAR', 'SINGLE_PION_END', 'SINGLE_PION_0510', 'PileUpDataMC', 'PileUpOOT', 'lumierr', 'TIME', 'PileUpBias', 'SINGLE_PION_1015', 'RelativeStatHF', 'RelativeJERC1', 'RelativeFSR', 'npcorerr']
    res_array = data.values()
    res = (np.vstack(res_array)).transpose()
    header = ' '.join(['{:>12s}'.format(k) for k in data.keys()])[2:]
    np.savetxt("H1JetsAtHighQ2HERAII_NormIncJetsQ2Pt_unfolding_prep.txt", res, header=header, fmt='%12.5g')
    
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

#!/usr/bin/env python2

import numpy, sys, os
import math

def main(args):
	data_file = "InclusiveJets_Table_postCWR.txt"
	stat_correlations_file = "InclusiveJets_StatCorrelations_postCWR.txt"
	outfile = "../qcd11004v2.npz"
	
	in_data = numpy.genfromtxt(data_file,dtype=float,skip_header=0,names=True)
	in_stat_correlations = numpy.genfromtxt(stat_correlations_file,dtype = float)

	npz_storage = {}

	relative = ['stat_low', 'lumierr', 'stat_high', 'uncor', 'npcorerr', 'unf_err_low', 'unf_err_high', 'ABSOLUTE_low', 'ABSOLUTE_high', 'High_PT_EXTR_low', 'High_PT_EXTR_high', 'SINGLE_PION_low', 'SINGLE_PION_high', 'FLAVOUR_low', 'FLAVOUR_high', 'TIME_low', 'TIME_high', 'RelativeJERC1_low', 'RelativeJERC1_high', 'RelativeJERC2_low', 'RelativeJERC2_high', 'RelativeJERCHF_low', 'RelativeJERCHF_high', 'RelativeStatEC2_low', 'RelativeStatEC2_high', 'RelativeStatHF_low', 'RelativeStatHF_high', 'RelativeFSR_low', 'RelativeFSR_high', 'PileUpDataMC_low', 'PileUpDataMC_high', 'PileUpOOT_low', 'PileUpOOT_high', 'PileUpPt_low', 'PileUpPt_high', 'PileUpBias_low', 'PileUpBias_high', 'PileUpJetRate_low', 'PileUpJetRate_high']

	for item in in_data.dtype.names:
		if item in relative:
			npz_storage[item] = in_data[item]*in_data['xs']
			continue
		npz_storage[item] = in_data[item]

	jec_names = {
	'jec_src_0' : 'ABSOLUTE',
	'jec_src_1'  : 'High_PT_EXTR',
	'jec_src_2'  : 'SINGLE_PION',
	'jec_src_3'  : 'FLAVOUR',
	'jec_src_4'  : 'TIME',
	'jec_src_5'  : 'RelativeJERC1',
	'jec_src_6'  : 'RelativeJERC2',
	'jec_src_7'  : 'RelativeJERCHF',
	'jec_src_8'  : 'RelativeStatEC2',
	'jec_src_9'  : 'RelativeStatHF',
	'jec_src_10' : 'RelativeFSR',
	'jec_src_11' : 'PileUpDataMC',
	'jec_src_12' : 'PileUpOOT',
	'jec_src_13' : 'PileUpPt',
	'jec_src_14' : 'PileUpBias',
	'jec_src_15' : 'PileUpJetRate',
	}
	
	#stat_cor_matrix = get_cor_matrix_from_table(in_stat_correlations, in_data["ylow"], in_data["ptlow"])
	#npz_storage['cor_stat'] = stat_cor_matrix
	
	sym_errors = ['ABSOLUTE_xxx', 'High_PT_EXTR_xxx', 'SINGLE_PION_xxx', 
					'FLAVOUR_xxx', 'TIME_xxx', 'RelativeJERC1_xxx', 
					'RelativeJERC2_xxx', 'RelativeJERCHF_xxx', 
					'RelativeStatEC2_xxx', 'RelativeStatHF_xxx', 
					'RelativeFSR_xxx', 'PileUpDataMC_xxx', 'PileUpOOT_xxx', 
					'PileUpPt_xxx', 'PileUpBias_xxx', 'PileUpJetRate_xxx',  
					'stat_xxx','unf_err_xxx']
	
	for unc_source in sym_errors:
		#unc = get_max_from_abs(in_data[unc_source.replace('xxx','low')],in_data[unc_source.replace('xxx','high')])
		#unc = ((1 + in_data[unc_source.replace('xxx','high')]) - (1. -in_data[unc_source.replace('xxx','low')]))/2.
		unc = (numpy.abs(npz_storage[unc_source.replace('xxx','high')]) + numpy.abs(npz_storage[unc_source.replace('xxx','low')]))/2.
		npz_storage[unc_source.replace('_xxx','')] = unc


	npz_storage['SINGLE_PION_BAR'] = npz_storage['SINGLE_PION'].copy / math.sqrt(2)
	npz_storage['SINGLE_PION_BAR'][npz_storage['ylow'] >  1.0] = 0.0
	npz_storage['SINGLE_PION_END'] = npz_storage['SINGLE_PION'].copy
    npz_storage['SINGLE_PION_END'][npz_storage['ylow'] <  1.5] = 0.0
	npz_storage['SINGLE_PION_0005'] = npz_storage['SINGLE_PION'].copy / math.sqrt(2)
	npz_storage['SINGLE_PION_0005'][npz_storage['ylow'] !=  0.0] = 0.0
	npz_storage['SINGLE_PION_0510'] = npz_storage['SINGLE_PION'].copy / math.sqrt(2)
	npz_storage['SINGLE_PION_0510'][npz_storage['ylow'] !=  0.5] = 0.0
	npz_storage['SINGLE_PION_1015'] = npz_storage['SINGLE_PION'].copy / math.sqrt(2)
	npz_storage['SINGLE_PION_1015'][npz_storage['ylow'] !=  1.0] = 0.0

	del npz_storage['SINGLE_PION']

	#numpy.savez(outfile, **npz_storage)
	# conatenate arrays
	ordered_list = ['ylow', 'yhigh', 'ptlow', 'pthigh', 'xs', 'stat', 'uncor', 'High_PT_EXTR', 'FLAVOUR', 'RelativeStatEC2', 'RelativeJERCHF', 'SINGLE_PION_0005', 'RelativeJERC2', 'ABSOLUTE', 'PileUpJetRate', 'unf_err', 'PileUpPt', 'npcor', 'SINGLE_PION_BAR', 'SINGLE_PION_END', 'SINGLE_PION_0510', 'PileUpDataMC', 'PileUpOOT', 'lumierr', 'TIME', 'PileUpBias', 'SINGLE_PION_1015', 'RelativeStatHF', 'RelativeJERC1', 'RelativeFSR', 'npcorerr']
	res_array = [npz_storage[key] for key in ordered_list]
	res = (numpy.vstack(res_array)).transpose()
	header = ' '.join(['{:>20s}'.format(k) for k in ordered_list])[2:]
	numpy.savetxt('CMS_Inclusive_Jets_2011.txt', res, header=header, fmt='%20.5g')
	
def get_max_from_abs(np1,np2):
	res = numpy.zeros(np1.shape)
	for i in range(0,len(np1)):
		res[i] = np1[i] if (numpy.abs(np1[i]) > numpy.abs(np2[i])) else np2[i]
	return res

def get_cor_matrix_from_table(correlation_table, bin_1, bin_2):
	cor_matrix = numpy.zeros([len(bin_1),len(bin_1)])
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
	cov_matrix = numpy.zeros([len(stat),len(stat)])
	for i in range(0,len(stat)):
		for j in range(0,len(stat)):
			cov_matrix[i][j] = correlation_matrix[i][j]*stat[i]*stat[j] 
	return cov_matrix


def get_cor_from_syst(syst):
	cor_matrix = numpy.zeros([len(syst),len(syst)])
	for i in range(0,len(syst)):
		for j in range(0,len(syst)):
			cor_matrix[i][j] = syst[i] * syst[j]
	return cor_matrix

def get_cor_from_uncor(uncor):
	cor_matrix = numpy.zeros([len(uncor),len(uncor)])
	for i in range(0,len(uncor)):
		cor_matrix[i][i] = uncor[i]*uncor[i]
	return cor_matrix

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

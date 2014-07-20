#!/usr/bin/env python2

import os
import sys
import numpy


def main(args):
	corrs = numpy.genfromtxt('InclusiveJets_StatCorrelations_postCWR.txt', names=True)
	data = numpy.genfromtxt('InclusiveJets_Table_postCWR.txt', names=True)

	print data['y_lo']
	print data['pt_lo']

	corr_matrix = get_corr_matrix_from_table(corrs, data['y_lo'], data['pt_lo'])
	print corr_matrix
	numpy.savetxt("CMS_InclusiveJets_2011_statcorrelations.txt",corr_matrix, fmt='%4.2f',delimiter=' ')

def get_corr_from_cov(cov_matrix):
	corr_matrix = numpy.zeros(cov_matrix.shape)
	for j in range(0,cov_matrix.shape[0]):
		for k in range(0, cov_matrix.shape[1]):
			corr_matrix[j][k] =  cov_matrix[j][k] / (numpy.sqrt(cov_matrix[j][j]) * numpy.sqrt(cov_matrix[k][k]))
	return corr_matrix

def get_corr_matrix_from_table(correlation_table, bin_1, bin_2):
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

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

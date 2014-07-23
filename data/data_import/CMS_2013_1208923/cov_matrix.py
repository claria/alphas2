#!/usr/bin/env python2

import os
import sys
import numpy


def main(args):
	cov_matrix = numpy.loadtxt("covariance_matrix_dijets.txt")
	print cov_matrix.shape
	print cov_matrix
	corr_matrix = get_corr_from_cov(cov_matrix)
	print corr_matrix
	numpy.savetxt("correlation_matrix_dijets.txt",corr_matrix, fmt='%.2f',delimiter=' ')

def get_corr_from_cov(cov_matrix):
	corr_matrix = numpy.zeros(cov_matrix.shape)
	for j in range(0,cov_matrix.shape[0]):
		for k in range(0, cov_matrix.shape[1]):
			corr_matrix[j][k] =  cov_matrix[j][k] / (numpy.sqrt(cov_matrix[j][j]) * numpy.sqrt(cov_matrix[k][k]))
	return corr_matrix

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

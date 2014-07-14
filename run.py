#!/usr/bin/env python2

import sys
import argparse
import logging

#import numpy
from src.alphas import calculate_chi2, perform_fit, plot

# Initialize logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
logger = logging.getLogger()


def main():

    # Parse args
    logger.debug('Parsing command line args')
    ##########
    # Parent #
    ##########

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-c', '--config', help='Analysis config')

    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers()
    ###########
    # Chi2 #
    ###########

    parser_chi2 = subparsers.add_parser('chi2', help='Calculate Chi2', parents=[parent_parser])
    parser_chi2.add_argument('-p', '--pdfset', type=str, help='PDF set to be used.')
    parser_chi2.add_argument('-a', '--asmz', type=float, default=0.1184, help='AlphasMz value to be used in calculation')
    parser_chi2.add_argument('-d', '--datasets', type=str, nargs='+', help='Datasets to be used')
    parser_chi2.set_defaults(func=calculate_chi2)

    ###########
    # Fitting #
    ###########

    parser_fit = subparsers.add_parser('fit', help='Do the Fit', parents=[parent_parser])
    parser_fit.add_argument('-p', '--pdfset', type=str, help='PDF set to use in the fit')
    parser_fit.add_argument('-d', '--datasets', type=str, nargs='+', help='Datasets to use in fits')
    parser_fit.set_defaults(func=perform_fit)

    ############
    # Plotting #
    ############

    parser_plot = subparsers.add_parser('plot', help='Do the plotting', parents=[parent_parser])
    parser_plot.add_argument('plot', type=str, choices=['running'], help='Types of plots')
    parser_plot.set_defaults(func=plot)

    # Save all commandline arguments in dict
    kwargs = vars(parser.parse_args())

    # Call desired function with all supplied keyword arguments
    if kwargs['func']:
        kwargs['func'](**kwargs)


if __name__ == '__main__':
    sys.exit(main())

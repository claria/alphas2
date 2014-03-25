#!/usr/bin/env python2

import sys
import argparse
import logging

#import numpy
from src.fitter import perform_fit, plot, profile_likelihood


# Initialize logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger()


def main():
    # Parse args
    logger.info('Parsing command line args')

    # Parent
    parent_parser = argparse.ArgumentParser(add_help=False, prog='PROG')
    parent_parser.add_argument('-c', '--config', help='Analysis config', required=True)
    # Fitting
    parser = argparse.ArgumentParser(add_help=False, prog='PROG')
    subparsers = parser.add_subparsers()
    parser_fit = subparsers.add_parser('fit', help='Do the Fit',
                                       parents=[parent_parser])
    parser_fit.set_defaults(func=perform_fit)

    parser_profile = subparsers.add_parser('profile', help='Do the Fit',
                                           parents=[parent_parser])
    parser_profile.set_defaults(func=profile_likelihood)
    # Plotting
    parser_plot = subparsers.add_parser('plot', help='Do the plotting',
                                        parents=[parent_parser])
    parser_plot.add_argument('plot', type=str,
                             choices=['running'],
                             help='Types of plots')
    parser_plot.set_defaults(func=plot)

    # Save all commandline arguments in dict
    kwargs = vars(parser.parse_args())

    logger.debug("Command line arguments:")
    logger.debug(str(kwargs))

    # Call desired function with all supplied keyword arguments
    kwargs['func'](**kwargs)


if __name__ == '__main__':
    sys.exit(main())

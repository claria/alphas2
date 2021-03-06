import os

import numpy as np


class ArrayDict(dict):
    """
    ArrayDict objects are dictionaries containing named np arrays
    """

    def __init__(self, from_file=None, **kwargs):
        """
        an ArrayDict can be created either empty 
            x = ArrayDict()
        or filled with content from a npz file
            x = ArrayDict('file.npz')
        or merge content from several npz files
        (if there's a * or ? in the filename)
            x = ArrayDict('dir/*.npz')
        or from variables
            x = ArrayDict(lon=lon, lat=lat)
        """
        dict.__init__(self)

        if from_file:

            if ('?' in from_file) or ('*' in from_file):
                # if from_file looks like a file pattern

                import glob

                filelist = glob.glob(from_file)
                if not filelist:
                    print 'no file matching pattern', from_file
                    return

                filelist.sort()
                print 'Aggregating data from %d files' % len(filelist)

                for f in filelist:
                    try:
                        this_array = ArrayDict(from_file=f)
                    except:
                        continue
                    self.append(this_array)

            else:

                npz = np.load(from_file)
                for f in npz.files:
                    self[f] = npz[f]
                npz.close()

        if kwargs:
            for key in kwargs:
                self[key] = kwargs[key]

    def append(self, arraydict, axis=0):
        """
        Appends np arrays contained in another arraydict with those present in self.
        0-d arrays are ignored.
        """

        arrnames = arraydict.keys()
        if arrnames == []:
            return

        for arrname in arrnames:
            if np.shape(arraydict[arrname]) is ():
                continue
            if arrname in self.keys():
                self[arrname] = np.concatenate(
                    (self[arrname], arraydict[arrname]), axis=axis)
            else:
                self[arrname] = arraydict[arrname]

    def list(self):
        """
        display the list of arrays contained in self and their shape
        """

        for arrname in self.keys():
            print arrname, ':', self[arrname].shape

    def save(self, filename, verbose=True):
        """
        save the arrays in a np file
        """
        if verbose:
            print 'Saving', filename
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.savez(filename, **self)

    def dump(self, filename):
        """
        save the arrays in a np file
        """
        np.savez(filename, **self)

    def get_vars(self, varnamelist):
        """
        input: a list containing names of variables
        returns: a list containing the associated variables
        """
        varlist = []
        for varname in varnamelist:
            varlist.append(self[varname])
        return varlist

    def subset(self, idx):
        """
        filters out the contained variables along their first dimension according to an index vector
        the index vector must have the same number of items in the first dimension as every variable in the arraydict
        """

        for arrname in self.keys():
            self[arrname] = self[arrname][idx, ...]


#class arraydict(ArrayDict):
#   pass


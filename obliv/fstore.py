"""Module for a file-based backend for vORAM storage.

The fstore class stores files in a given directory and provides
an interface similar to a Python list over those files.

This can be used for locally testing vORAMs or to provide
persistence. If the directory is, say, mounted to a remote
server, this can be used to provide a remove vORAM interface.
"""

import os
from os import path
from collections import MutableSequence, Counter


class fstore(MutableSequence):
    """Provides a list-like interface which stores bytes in files.
    
    Backend files are stored in a given directory with names like 132.fstore.

    All normal list operations and operators are supported, either directly
    or through the MutableSubsequence superclass.

    This class supports context management with the ``__enter__`` and 
    ``__exit__`` methods, so for example you can do:
        
        with fstore('mydir') as mystore:
            mystore[0] = b'some data'

    (Although there isn't actually any context to manage and those
    methods are no-op's right now.)
    """

    def __init__(self, dirname):
        """Creates a list view on storage files in the given directory."""
        self._basedir = path.abspath(dirname)

        self._count = Counter()

        if path.isdir(self.basedir):
            self._len = self._determine_len()
        elif path.exists(self.basedir):
            raise ValueError("Can't use directory {}; already exists."
                .format(self.basedir))
        else:
            os.mkdir(self.basedir)
            self._len = 0

    @property
    def basedir(self):
        """The path where all files are stored"""
        return self._basedir

    def update_base(self,basedir):
        """Changes the base directory without moving any files (dangerous!)."""
        self._basedir=basedir

    def __len__(self):
        return self._len

    def _ind(self, ind):
        """Computes the actual index and checks out of bounds."""
        posind = ind if ind >= 0 else len(self) + ind
        if 0 <= posind < len(self):
            return posind
        else:
            raise IndexError("index out of bounds: {}".format(ind))

    def __getitem__(self, ind):
        """Implements indexed lookup f[i]"""
        with open(self._getfile(self._ind(ind)), mode='rb') as blockf:
            res = blockf.read()
            self._count[len(res)]+=1
            return res

    def __setitem__(self, ind, value):
        """Implements indexed assignment f[i] = x"""
        assert isinstance(value, bytes)
        with open(self._getfile(self._ind(ind)), mode='wb') as blockf:
            assert blockf.write(value) == len(value)
            self._count[len(value)]+=1

    def __delitem__(self, ind):
        """Implements indexed deletion: del f[i].
        
        Only deletion from the end is actually supported."""
        actind = self._ind(ind)
        if actind != self._len - 1:
            raise IndexError("You can only remove from the end.")
        os.remove(self._getfile(self._ind(ind)))
        self._len -= 1

    def insert(self, ind, value):
        """Inserts value at index ind.
        
        Only insertion at the end is actually supported."""
        if ind == len(self):
            self._len += 1
            self.__setitem__(ind, value)
        else:
            raise IndexError("You can only insert at the end.")

    def _getfile(self, ind):
        """Generates the filename for the given index."""
        return path.join(self.basedir, str(ind) + ".fstore")

    def _determine_len(self):
        """Does a gallop search to determine the length based on actual
        files stored."""
        lower = 0
        upper = 1
        while path.exists(self._getfile(upper)):
            upper *= 2
        while lower < upper:
            mid = (lower + upper) // 2
            if path.exists(self._getfile(mid)):
                lower = mid + 1
            else:
                upper = mid
        return lower

    def __enter__(self):
        return self
        #pass

    def __exit__(self, extype, excal, tb):
        pass

    def clear_counts(self):
        """Resets the file size counts."""
        self._count.clear()

    def print_counts(self):
        """For debugging purposes, prints out the statistics of how many files
        were transfered of what sizes."""
        if len(self._count) > 10:
            print("More than 10 different block sizes; THERE IS A PROBLEM!!!")
        else:
            for bs in sorted(self._count):
                print("Transfered {} files of size {}".format(self._count[bs], bs))
            if not self._count:
                print("ZERO transfers performed.")
